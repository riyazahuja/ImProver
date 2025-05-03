/-- The successor of `a : WithBot α` as an element of `α`. -/
def succ (a : WithBot α) : α := a.recBotCoe ⊥ Order.succ


/-- Not to be confused with `WithBot.orderSucc_bot`, which is about `Order.succ`. -/
@[simp] lemma succ_bot : succ (⊥ : WithBot α) = ⊥ := rfl


/-- Not to be confused with `WithBot.orderSucc_coe`, which is about `Order.succ`. -/
@[simp] lemma succ_coe (a : α) : succ (a : WithBot α) = Order.succ a := rfl


lemma succ_eq_succ : ∀ a : WithBot α, succ a = Order.succ a
  | ⊥ => rfl
  | (a : α) => rfl


lemma succ_mono : Monotone (succ : WithBot α → α)
                  /-
                    α : Type u_1
                    inst✝² : Preorder α
                    inst✝¹ : OrderBot α
                    inst✝ : SuccOrder α
                    x✝¹ : WithBot α
                    x✝ : LE.le Bot.bot x✝¹
                    ⊢ LE.le Bot.bot.succ x✝¹.succ
                  -/
  | ⊥, _, _ => by simp
                  /-
                    🎉 no goals
                  -/
                          /-
                            α : Type u_1
                            inst✝² : Preorder α
                            inst✝¹ : OrderBot α
                            inst✝ : SuccOrder α
                            a : α
                            hab : LE.le (↑a) Bot.bot
                            ⊢ LE.le (↑a).succ Bot.bot.succ
                          -/
  | (a : α), ⊥, hab => by simp at hab
                          /-
                            🎉 no goals
                          -/
                                                    /-
                                                      α : Type u_1
                                                      inst✝² : Preorder α
                                                      inst✝¹ : OrderBot α
                                                      inst✝ : SuccOrder α
                                                      a b : α
                                                      hab : LE.le ↑a ↑b
                                                      ⊢ LE.le a b
                                                    -/
  | (a : α), (b : α), hab => Order.succ_le_succ (by simpa using hab)
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma succ_strictMono [NoMaxOrder α] : StrictMono (succ : WithBot α → α)
                          /-
                            α : Type u_1
                            inst✝³ : Preorder α
                            inst✝² : OrderBot α
                            inst✝¹ : SuccOrder α
                            inst✝ : NoMaxOrder α
                            b : α
                            hab : LT.lt Bot.bot ↑b
                            ⊢ LT.lt Bot.bot.succ (↑b).succ
                          -/
  | ⊥, (b : α), hab => by simp
                          /-
                            🎉 no goals
                          -/
                                                    /-
                                                      α : Type u_1
                                                      inst✝³ : Preorder α
                                                      inst✝² : OrderBot α
                                                      inst✝¹ : SuccOrder α
                                                      inst✝ : NoMaxOrder α
                                                      a b : α
                                                      hab : LT.lt ↑a ↑b
                                                      ⊢ LT.lt a b
                                                    -/
  | (a : α), (b : α), hab => Order.succ_lt_succ (by simpa using hab)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[gcongr] lemma succ_le_succ (hxy : x ≤ y) : x.succ ≤ y.succ := succ_mono hxy

@[gcongr] lemma succ_lt_succ [NoMaxOrder α] (hxy : x < y) : x.succ < y.succ := succ_strictMono hxy


/-- The predecessor of `a : WithTop α` as an element of `α`. -/
def pred (a : WithTop α) : α := a.recTopCoe ⊤ Order.pred


/-- Not to be confused with `WithTop.orderPred_top`, which is about `Order.pred`. -/
@[simp] lemma pred_top : pred (⊤ : WithTop α) = ⊤ := rfl


/-- Not to be confused with `WithTop.orderPred_coe`, which is about `Order.pred`. -/
@[simp] lemma pred_coe (a : α) : pred (a : WithTop α) = Order.pred a := rfl


lemma pred_eq_pred : ∀ a : WithTop α, pred a = Order.pred a
  | ⊤ => rfl
  | (a : α) => rfl


lemma pred_mono : Monotone (pred : WithTop α → α)
                  /-
                    α : Type u_1
                    inst✝² : Preorder α
                    inst✝¹ : OrderTop α
                    inst✝ : PredOrder α
                    x✝¹ : WithTop α
                    x✝ : LE.le x✝¹ Top.top
                    ⊢ LE.le x✝¹.pred Top.top.pred
                  -/
  | _, ⊤, _ => by simp
                  /-
                    🎉 no goals
                  -/
                          /-
                            α : Type u_1
                            inst✝² : Preorder α
                            inst✝¹ : OrderTop α
                            inst✝ : PredOrder α
                            a : α
                            hab : LE.le Top.top ↑a
                            ⊢ LE.le Top.top.pred (↑a).pred
                          -/
  | ⊤, (a : α), hab => by simp at hab
                          /-
                            🎉 no goals
                          -/
                                                    /-
                                                      α : Type u_1
                                                      inst✝² : Preorder α
                                                      inst✝¹ : OrderTop α
                                                      inst✝ : PredOrder α
                                                      a b : α
                                                      hab : LE.le ↑a ↑b
                                                      ⊢ LE.le a b
                                                    -/
  | (a : α), (b : α), hab => Order.pred_le_pred (by simpa using hab)
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma pred_strictMono [NoMinOrder α] : StrictMono (pred : WithTop α → α)
                          /-
                            α : Type u_1
                            inst✝³ : Preorder α
                            inst✝² : OrderTop α
                            inst✝¹ : PredOrder α
                            inst✝ : NoMinOrder α
                            b : α
                            hab : LT.lt (↑b) Top.top
                            ⊢ LT.lt (↑b).pred Top.top.pred
                          -/
  | (b : α), ⊤, hab => by simp
                          /-
                            🎉 no goals
                          -/
                                                    /-
                                                      α : Type u_1
                                                      inst✝³ : Preorder α
                                                      inst✝² : OrderTop α
                                                      inst✝¹ : PredOrder α
                                                      inst✝ : NoMinOrder α
                                                      a b : α
                                                      hab : LT.lt ↑a ↑b
                                                      ⊢ LT.lt a b
                                                    -/
  | (a : α), (b : α), hab => Order.pred_lt_pred (by simpa using hab)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[gcongr] lemma pred_le_pred (hxy : x ≤ y) : x.pred ≤ y.pred := pred_mono hxy

@[gcongr] lemma pred_lt_pred [NoMinOrder α] (hxy : x < y) : x.pred < y.pred := pred_strictMono hxy


