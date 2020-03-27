import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from _function import MyFunc


class TextImage(MyFunc):
    def __init__(self, font='Calibri/calibri.ttf'):
        super().__init__()
        self.use_font = "C:/Windows/Fonts" + font
        pass

    def main(self, true, estimate, out_put='test'):
        # 使うフォント，サイズ，描くテキストの設定
        fontsize = 36
        text1 = "True           : " + str(true)
        text2 = "Estimated : " + str(estimate)

        # 画像サイズ，背景色，フォントの色を設定
        canvasSize = (500, 200)
        backgroundRGB = (0, 0, 0)
        textRGB = (255, 255, 255)

        # 文字を描く画像の作成
        img = PIL.Image.new('RGB', canvasSize, backgroundRGB)
        draw = PIL.ImageDraw.Draw(img)

        # 用意した画像に文字列を描く
        font = PIL.ImageFont.truetype(self.use_font, fontsize)
        text1Width, text1Height = draw.textsize(text1, font=font)
        text2Width, text2Height = draw.textsize(text2, font=font)
        # print(text1Height, text2Height)
        # print(canvasSize[0]//3-text1Height//2)
        # print(canvasSize[0]//3 * 2-text2Height//2)
        text1TopLeft = (canvasSize[0]//6, canvasSize[1]//3-text1Height//2)  # 前から1/6，上下中央に配置
        text2TopLeft = (canvasSize[0]//6, canvasSize[1]//3 * 2-text2Height//2)  # 前から1/6，上下中央に配置
        draw.text(text1TopLeft, text1, fill=textRGB, font=font)
        draw.text(text2TopLeft, text2, fill=textRGB, font=font)

        path = self.make_dir_path(img=True, directory_name='/text/')
        img.save(path + out_put + ".png")


if __name__ == '__main__':
    TextImage().main(-40, -40.45)
