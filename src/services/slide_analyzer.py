# services/slide_analyzer.py

from fastapi import HTTPException
from api.dependencies import gemini_model
from utils.pdf_utils import extract_text_and_font_size, analyze_font_metrics

class SlideAnalyzer:
    """Analyze slide content."""
    def __init__(self):
        self.content_categories = {
            "technical": "技術的な説明や実装の詳細",
            "background": "背景説明や課題提起",
            "methodology": "手法や解決アプローチの説明",
            "results": "結果や成果の説明",
            "conclusion": "まとめや今後の展望"
        }

    def analyze_slide(self, pdf_data: bytes) -> str:
        text_blocks = extract_text_and_font_size(pdf_data)
        font_sizes = [size for _, size in text_blocks]
        font_analysis = analyze_font_metrics(font_sizes)
        
        full_text = " ".join([text for text, _ in text_blocks])
        
        prompt = (
            f"スライドの分析を行ってください。以下の点について詳細に評価してください：\n"
            f"1. 内容の一貫性\n"
            f"2. 視覚的一貫性\n"
            f"   - 平均フォントサイズ: {font_analysis['mean_size']:.1f}\n"
            f"   - フォントサイズのばらつき: {font_analysis['std_size']:.1f}\n"
            f"   - 使用されているフォントサイズの種類: {font_analysis['size_variation']}個\n"
            f"3. メインメッセージ\n\n"
            f"スライド内容:\n{full_text}"
        )

        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Geminiエラー: {str(e)}")