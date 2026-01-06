using System;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;
using System.Threading;

public class HandDetection : MonoBehaviour
{
    public HandPreview handPreview;
    public ImagePreview imagePreview;
    public ModelAsset handDetector;
    public ModelAsset handLandmarker;
    public TextAsset anchorsCSV;

    public float scoreThreshold = 0.5f;

    private WebCamTexture m_WebCamTexture;
    private CancellationTokenSource m_Cts; // 직접 관리할 취소 토큰 소스

    const int k_NumAnchors = 2016;
    float[,] m_Anchors;

    const int k_NumKeypoints = 21;
    const int detectorInputSize = 192;
    const int landmarkerInputSize = 224;

    Worker m_HandDetectorWorker;
    Worker m_HandLandmarkerWorker;
    Tensor<float> m_DetectorInput;
    Tensor<float> m_LandmarkerInput;

    float m_TextureWidth;
    float m_TextureHeight;

    public async void Start()
    {
        // 1. 취소 토큰 초기화
        m_Cts = new CancellationTokenSource();
        var token = m_Cts.Token;

        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        // 모델 로드 및 후처리 (기존 로직 유지)
        var handDetectorModel = ModelLoader.Load(handDetector);
        var graph = new FunctionalGraph();
        var input = graph.AddInput(handDetectorModel, 0);
        var outputs = Functional.Forward(handDetectorModel, input);
        var boxes = outputs[0];
        var scores = outputs[1];
        var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
        handDetectorModel = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        m_HandDetectorWorker = new Worker(handDetectorModel, BackendType.GPUCompute);

        var handLandmarkerModel = ModelLoader.Load(handLandmarker);
        m_HandLandmarkerWorker = new Worker(handLandmarkerModel, BackendType.GPUCompute);

        m_DetectorInput = new Tensor<float>(new TensorShape(1, detectorInputSize, detectorInputSize, 3));
        m_LandmarkerInput = new Tensor<float>(new TensorShape(1, landmarkerInputSize, landmarkerInputSize, 3));

        // 2. 웹캠 초기화
        if (WebCamTexture.devices.Length > 0)
        {
            m_WebCamTexture = new WebCamTexture(WebCamTexture.devices[0].name, 640, 480, 30);
            m_WebCamTexture.Play();
        }
        else
        {
            Debug.LogError("카메라를 찾을 수 없습니다.");
            return;
        }

        // 3. 실시간 추적 루프
        while (!token.IsCancellationRequested)
        {
            try
            {
                if (m_WebCamTexture != null && m_WebCamTexture.didUpdateThisFrame)
                {
                    await Detect(m_WebCamTexture, token);
                }
                else
                {
                    // 유니티 6의 Awaitable에 토큰 전달
                    await Awaitable.NextFrameAsync(token);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error during detection: {e.Message}");
                break;
            }
        }

        // 루프 종료 후 자원 정리
        Cleanup();
    }

    async Awaitable Detect(Texture texture, CancellationToken token)
    {
        if (token.IsCancellationRequested || imagePreview == null || handPreview == null) return;

        m_TextureWidth = texture.width;
        m_TextureHeight = texture.height;
        imagePreview.SetTexture(texture);

        var size = Mathf.Max(texture.width, texture.height);
        var scale = size / (float)detectorInputSize;
        var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

        m_HandDetectorWorker.Schedule(m_DetectorInput);

        var outputIdxAwaitable = (m_HandDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoreAwaitable = (m_HandDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxAwaitable = (m_HandDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIdx = await outputIdxAwaitable;
        using var outputScore = await outputScoreAwaitable;
        using var outputBox = await outputBoxAwaitable;

        // 대기 후 오브젝트 생존 확인
        if (token.IsCancellationRequested || handPreview == null) return;

        var scorePassesThreshold = outputScore[0] >= scoreThreshold;
        handPreview.SetActive(scorePassesThreshold);

        if (!scorePassesThreshold) return;

        var idx = outputIdx[0];
        var anchorPosition = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);
        var boxCentre_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 0], outputBox[0, 0, 1]);
        var boxSize_TensorSpace = math.max(outputBox[0, 0, 2], outputBox[0, 0, 3]);

        var kp0_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 0 + 0], outputBox[0, 0, 4 + 2 * 0 + 1]);
        var kp2_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 2 + 0], outputBox[0, 0, 4 + 2 * 2 + 1]);
        var delta_TensorSpace = kp2_TensorSpace - kp0_TensorSpace;
        var up_TensorSpace = delta_TensorSpace / math.length(delta_TensorSpace);
        var theta = math.atan2(delta_TensorSpace.y, delta_TensorSpace.x);
        var rotation = 0.5f * Mathf.PI - theta;
        boxCentre_TensorSpace += 0.5f * boxSize_TensorSpace * up_TensorSpace;
        boxSize_TensorSpace *= 2.6f;

        var origin2 = new float2(0.5f * landmarkerInputSize, 0.5f * landmarkerInputSize);
        var scale2 = boxSize_TensorSpace / landmarkerInputSize;
        var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(boxCentre_TensorSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(rotation)), BlazeUtils.TranslationMatrix(-origin2)));
        BlazeUtils.SampleImageAffine(texture, m_LandmarkerInput, M2);

        m_HandLandmarkerWorker.Schedule(m_LandmarkerInput);

        var landmarksAwaitable = (m_HandLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
        using var landmarks = await landmarksAwaitable;

        // 최종 렌더링 전 확인
        if (token.IsCancellationRequested || handPreview == null) return;

        for (var i = 0; i < k_NumKeypoints; i++)
        {
            if (handPreview == null) break;
            var position_ImageSpace = BlazeUtils.mul(M2, new float2(landmarks[3 * i + 0], landmarks[3 * i + 1]));
            Vector3 position_WorldSpace = ImageToWorld(position_ImageSpace) + new Vector3(0, 0, landmarks[3 * i + 2] / m_TextureHeight);
            handPreview.SetKeypoint(i, true, position_WorldSpace);
        }
    }

    Vector3 ImageToWorld(Vector2 position)
    {
        return (position - 0.5f * new Vector2(m_TextureWidth, m_TextureHeight)) / m_TextureHeight;
    }

    private void Cleanup()
    {
        m_HandDetectorWorker?.Dispose();
        m_HandLandmarkerWorker?.Dispose();
        m_DetectorInput?.Dispose();
        m_LandmarkerInput?.Dispose();

        if (m_WebCamTexture != null)
        {
            m_WebCamTexture.Stop();
            Destroy(m_WebCamTexture);
            m_WebCamTexture = null;
        }
        Debug.Log("Resources Cleaned Up.");
    }

    void OnDestroy()
    {
        // 4. 파괴 시 즉각 취소 신호 발생
        if (m_Cts != null)
        {
            m_Cts.Cancel();
            m_Cts.Dispose();
        }
    }
}
