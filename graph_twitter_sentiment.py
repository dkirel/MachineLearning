from matplotlib import pyplot as plt, animation, style

def graph_sentiment(i):
    data = open('twitter_sentiment.txt', 'r').read()
    sentiments = data.split('\n')

    x = range(len(sentiments))

    y_curr = 0
    y = []
    
    for s in sentiments:
        if s == 'pos':
            y_curr += 1
        elif s == 'neg':
            y_curr -= 0.25

        y.append(y_curr)

    ax1.clear()
    ax1.plot(x, y)

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

sentiment_animation = animation.FuncAnimation(fig, graph_sentiment, interval=1000)
plt.show()

