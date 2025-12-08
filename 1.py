import pandas as pd
from functools import lru_cache

# Параметры задачи (д.е.)
init_cb1 = 100    # ЦБ1
init_cb2 = 800    # ЦБ2
init_dep = 400    # депозит
init_cash = 600   # свободные средства

# Пакеты (четверть первоначальной стоимости)
pack_cb1 = init_cb1 // 4   # 25
pack_cb2 = init_cb2 // 4   # 200
pack_dep = init_dep // 4   # 100

unit = pack_cb1  # базовая единица = 25 д.е.

def to_units(amount):
    return amount // unit

def from_units(units):
    return units * unit

# Начальные состояния в unit'ах
h1_0 = to_units(init_cb1)   # 4
h2_0 = to_units(init_cb2)   # 32
dep_0 = to_units(init_dep)  # 16
cash_0 = to_units(init_cash) # 24

# Размеры пакетов в unit'ах
p1_u = to_units(pack_cb1)   # 1 unit (ЦБ1)
p2_u = to_units(pack_cb2)   # 8 units (ЦБ2)
pd_u = to_units(pack_dep)   # 4 units (депозит)

# Сценарии по этапам: (вероятность, r_cb1, r_cb2, r_dep)
stages = [
    [ (0.60, 1.20, 1.10, 1.07),
      (0.30, 1.05, 1.02, 1.03),
      (0.10, 0.80, 0.95, 1.00) ],
    [ (0.30, 1.40, 1.15, 1.01),
      (0.20, 1.05, 1.00, 1.00),
      (0.50, 0.60, 0.90, 1.00) ],
    [ (0.40, 1.15, 1.12, 1.05),
      (0.40, 1.05, 1.01, 1.01),
      (0.20, 0.70, 0.94, 1.00) ]
]

# Функция генерации допустимых действий
def actions_for_state(state):
    """
    Возвращает список действий (d1, d2, dd) в unit'ах.
    Ограничены верхние значения покупки для сжатия пространства действий.
    d>0 означает покупку (трату cash), d<0 — продажу (пополнение cash).
    """
    h1, h2, dep, cash = state
    actions = []

    # Ограничения на покупки (для практичности; можно увеличить)
    max_buy1 = min(cash // p1_u, 4)    # максимум купить до 4 units ЦБ1 (<= 100 д.е.)
    max_buy2 = min(cash // p2_u, 1)    # максимум 1 пакет ЦБ2 (200 д.е.)
    max_buyd = min(cash // pd_u, 2)    # максимум 2 пакета депозита (<= 200 д.е.)

    # Ограничения на продажи (не больше имеющихся)
    min_d1 = -min(h1, 8)   # можно продать до 8 unit'ов (если есть)
    max_d1 = max_buy1
    min_d2 = -min(h2, p2_u*1)  # можно продать до 1 пакета CB2
    max_d2 = max_buy2 * p2_u
    min_dd = -min(dep, pd_u*2) # можно продать до 2 пакетов депозита
    max_dd = max_buyd * pd_u

    for d1 in range(min_d1, max_d1+1):
        for d2 in range(min_d2, max_d2+1, p2_u):
            for dd in range(min_dd, max_dd+1, pd_u):
                new_cash = cash - (d1 + d2 + dd)
                if new_cash < 0:
                    continue
                if h1 + d1 < 0 or h2 + d2 < 0 or dep + dd < 0:
                    continue
                actions.append((d1, d2, dd))
    return actions

# Оценка стоимости состояния
def state_value_de(state):
    h1, h2, dep, cash = state
    return from_units(h1) + from_units(h2) + from_units(dep) + from_units(cash)

# DP: Bellman recursion с мемоизацией
@lru_cache(maxsize=None)
def V(t, h1, h2, dep, cash):
    """
    Возвращает кортеж (оптимальное ожидаемое значение с этапа t, оптимальное действие).
    Состояния и действия в unit'ах (целые).
    """
    state = (h1, h2, dep, cash)
    if t == 3:
        return state_value_de(state), None

    best_value = -1e18
    best_action = None
    acts = actions_for_state(state)
    if not acts:
        acts = [(0,0,0)]

    for a in acts:
        expected = 0.0
        for p, r1, r2, rd in stages[t]:
            # Точность: считаем в д.е., затем переводим обратно в unit'ы (округлённо)
            h1_de = from_units(h1); h2_de = from_units(h2)
            dep_de = from_units(dep); cash_de = from_units(cash)
            d1_de = from_units(a[0]); d2_de = from_units(a[1]); dd_de = from_units(a[2])

            new_h1_de = (h1_de + d1_de) * r1
            new_h2_de = (h2_de + d2_de) * r2
            new_dep_de = (dep_de + dd_de) * rd
            new_cash_de = cash_de - (d1_de + d2_de + dd_de)

            # Переводим в unit'ы для состояния следующего шага
            next_h1 = int(round(new_h1_de / unit))
            next_h2 = int(round(new_h2_de / unit))
            next_dep = int(round(new_dep_de / unit))
            next_cash = int(round(new_cash_de / unit))

            val_next, _ = V(t+1, next_h1, next_h2, next_dep, next_cash)
            expected += p * val_next

        if expected > best_value:
            best_value = expected
            best_action = a

    return best_value, best_action

# Запуск решения
if __name__ == "__main__":
    opt_value, opt_action = V(0, h1_0, h2_0, dep_0, cash_0)

    # Построим демонстрационную траекторию, применяя оптимальные действия и наиболее вероятные сценарии
    trajectory = []
    t = 0
    state = (h1_0, h2_0, dep_0, cash_0)
    while t < 3:
        val, action = V(t, *state)
        trajectory.append((t, state, val, action))
        # выбрать наиболее вероятный сценарий для демонстрации пути
        scen = max(stages[t], key=lambda x: x[0])
        p, r1, r2, rd = scen

        h1, h2, dep, cash = state
        a = action if action is not None else (0,0,0)
        h1_de = from_units(h1); h2_de = from_units(h2)
        dep_de = from_units(dep); cash_de = from_units(cash)
        d1_de = from_units(a[0]); d2_de = from_units(a[1]); dd_de = from_units(a[2])

        new_h1_de = (h1_de + d1_de) * r1
        new_h2_de = (h2_de + d2_de) * r2
        new_dep_de = (dep_de + dd_de) * rd
        new_cash_de = cash_de - (d1_de + d2_de + dd_de)

        next_state = (int(round(new_h1_de / unit)),
                      int(round(new_h2_de / unit)),
                      int(round(new_dep_de / unit)),
                      int(round(new_cash_de / unit)))

        state = next_state
        t += 1

    # Таблица результатов
    rows = []
    for rec in trajectory:
        t, st, val, a = rec
        h1, h2, dep, cash = st
        rows.append({
            'этап (t)': t+1,
            'холд ЦБ1 (д.е.)': from_units(h1),
            'холд ЦБ2 (д.е.)': from_units(h2),
            'холд депоз (д.е.)': from_units(dep),
            'наличн (д.е.)': from_units(cash),
            'опт. действие (д.е.)': (from_units(a[0]), from_units(a[1]), from_units(a[2])),
            'ожид. стоимость после решения (д.е.)': round(val, 2)
        })

    df = pd.DataFrame(rows)
    print("Оптимальная ожидаемая итоговая стоимость (д.е.):", round(opt_value,2))
    print("Оптимальное начальное действие (delta_CB1, delta_CB2, delta_Dep в д.е.):",
          (from_units(opt_action[0]), from_units(opt_action[1]), from_units(opt_action[2])))
    print("\nДемонстрационная траектория (перед каждым этапом):")
    print(df.to_string(index=False))
