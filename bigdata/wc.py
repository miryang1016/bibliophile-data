import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

colors = ['#C5D887', '#FFA644', '#FFD66C', '#FF7628']
color_index = 0

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    global color_index
    color = colors[color_index % len(colors)]
    color_index += 1
    return color

def generate_wordcloud(dic_word):
    
    font_path = './Pretendard-Regular.ttf'
    wordcloud = WordCloud(width=1000, height=360, background_color='white', font_path=font_path, color_func=color_func, max_words=25).generate_from_frequencies(dic_word)

    img = io.BytesIO()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    img.seek(0)
    
    return img