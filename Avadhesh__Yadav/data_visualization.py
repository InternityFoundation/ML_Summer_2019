import matplotlib.pyplot as plt
days=range(1,8)
Y_views=[534,689,258,401,724,689,358]

plt.scatter(days,Y_views)
plt.xlabel('day')
plt.ylabel('views')
plt.legend(loc='upper right')

plt.grid(True,linewidth=1,color='r',linestyle='--')
plt.title("days views for marketing")
plt.show()