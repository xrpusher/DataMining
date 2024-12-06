import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns  # Добавляем импорт seaborn
import requests
import logging
import context
import simulator

mpl.use('Qt5Agg')
sns.set_theme()  # Устанавливаем тему seaborn вместо plt.style.use('seaborn')

class Animate:
    def __init__(self, host: str, port: int, context: context.Context, simulator: simulator.Simulator, log: logging.Logger):
        self.headers = {'Content-Type': 'application/json'}
        self.url = 'http://' + host + ':' + str(port)
        self.context = context
        self.dist = self.context.max_price - self.context.min_price
        self.simulator = simulator
        self.log = log
        self.params = {'ctx': self.context.context_hash}
        self.fig, (self.ax_true, self.ax_points, self.ax_space) = plt.subplots(nrows=3, ncols=1)

    @staticmethod
    def parse_winning_curve_response(response) -> (list[float], list[float]):
        learned_probabilities = response['learned_probability']
        prices = []
        if learned_probabilities is None:
            return None, None
        for p in response['probs']:
            prices.append(p['price'])
        return prices, learned_probabilities

    @staticmethod
    def parse_quality(response) -> (list[float], list[float]):
        return response['price'], response['quality']

    @staticmethod
    def net_revenue(prices) -> list[float]:
        net_revenue = []
        m = prices[len(prices) - 1]
        for p in prices:
            net_revenue.append(m - p)
        return net_revenue

    @staticmethod
    def max_p(net_revenue, probabilities) -> (list[float], float, int):
        e = [i * j for i, j in zip(probabilities, net_revenue)]
        m_id = -1
        m_val = 0.0
        for i, v in enumerate(e):
            if v > m_val:
                m_val = v
                m_id = i
        return e, m_val, m_id

    def quantities(self):
        response = requests.get(url=self.url + '/space', params=self.params, headers=self.headers)
        if response.status_code != 200:
            return
        resp_json = response.json()
        levels = resp_json['level']
        self.ax_true.cla()
        self.ax_true.plot(self.simulator.auction.prices(), self.simulator.auction.curve,
                          color='blue', label='# true win prob')
        self.ax_true.plot(self.simulator.auction.prices(), self.simulator.auction.net_revenue(),
                          color='blue', label='# true net revenue', linestyle='--')
        self.ax_true.plot(self.simulator.auction.prices(), self.simulator.auction.expectations(),
                          color='green', label='# true expectations', linestyle='-.')
        self.ax_true.scatter([self.simulator.auction.optimal_price()], [0.0],
                             alpha=1.0, color='red', marker='x', s=100, label='# optimal price')
        self.ax_true.legend()

        self.ax_points.cla()
        self.ax_points.set_title("how far bid price from optimal one in %")
        # plot line between best and predicted values
        for i in range(len(self.simulator.d)):
            x_1 = min(self.simulator.d[i][0], self.simulator.d[i][1])
            x_2 = max(self.simulator.d[i][0], self.simulator.d[i][1])
            color = 'grey'
            line_style = 'dotted'
            text = f"Error {format((x_2 - x_1) / self.dist, '.2f')}%"
            if ((x_2 - x_1) / self.dist) > 0.2:
                color = 'red'
                line_style = 'solid'
                text = f"Opt dist: {format((x_2 - x_1), '.4f')}, Dist: {format(self.dist, '.4f')}"
                self.log.debug(f"Opt dist {format((x_2 - x_1), '.4f')}, Dist: {format(self.dist, '.4f')}")
                self.log.debug(f"range_min:{self.context.min_price}, "
                               f"range_max:{self.context.max_price}, "
                               f"true_best_price:{self.simulator.d[i][0]}, "
                               f"opt_price:{self.simulator.d[i][1]}, "
                               f"bid_price:{self.simulator.d[i][2]}")
            self.ax_points.hlines(y=i*0.1, xmin=x_1, xmax=x_2, color=color, linestyles=line_style)
            mid = (x_1 + x_2) / 2.0
            self.ax_points.text(x=mid, y=i*0.1 + 0.02, s=text, ha='center')
        self.ax_points.legend()

        color = ['yellow', 'green', 'blue', 'grey', 'red']
        self.ax_space.cla()
        self.ax_space.set_title("assembly of learned probabilities")
        for i in range(len(levels)):
            self.ax_space.plot(levels[i]['price'], levels[i]['pr'], color=color[i])
        self.ax_space.legend()


def animate_call(i, animate):
    animate.quantities()


def run_animate(host: str, port: int, context: context.Context, simulator: simulator.Simulator, log: logging.Logger):
    animate = Animate(host, port, context, simulator, log)
    # Plot every 5 sec
    global_any = FuncAnimation(animate.fig, animate_call, interval=5000, fargs=(animate,))
    plt.tight_layout()
    plt.show(block=True)
