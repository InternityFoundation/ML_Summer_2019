import matplotlib.pyplot as plt
days=range(1,8)
Y_views=[534,689,258,401,724,689,358]
F_views=[123,342,700,304,405,650,5500]
T_views=[302,209,176,415,824,389,3987]
plt.plot(days,Y_views)
plt.plot(days,F_views)
plt.plot(days,T_views)
plt.xlabel('day')
plt.ylabel('views')
plt.legend(loc='upper right')

plt.grid(True,linewidth=1,color='r',linestyle='--')
plt.title("five days views for marketing")
plt.show()