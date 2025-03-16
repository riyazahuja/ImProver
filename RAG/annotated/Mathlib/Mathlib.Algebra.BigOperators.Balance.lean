/-- The balancing of a function, namely the function minus its average. -/
def balance (f : ι → G) : ι → G := f - Function.const _ (𝔼 y, f y)


lemma balance_apply (f : ι → G) (x : ι) : balance f x = f x - 𝔼 y, f y := rfl


                                                           /-
                                                             ι : Type u_1
                                                             G : Type u_4
                                                             inst✝² : Fintype ι
                                                             inst✝¹ : AddCommGroup G
                                                             inst✝ : Module NNRat G
                                                             ⊢ Eq (Fintype.balance 0) 0
                                                           -/
@[simp] lemma balance_zero : balance (0 : ι → G) = 0 := by simp [balance]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp] lemma balance_add (f g : ι → G) : balance (f + g) = balance f + balance g := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup G
    inst✝ : Module NNRat G
    f g : ι → G
    ⊢ Eq (Fintype.balance (HAdd.hAdd f g)) (HAdd.hAdd (Fintype.balance f) (Fintype …
  -/
  simp only [balance, expect_add_distrib, ← const_add, add_sub_add_comm, Pi.add_apply]
  /-
    🎉 no goals
  -/


@[simp] lemma balance_sub (f g : ι → G) : balance (f - g) = balance f - balance g := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup G
    inst✝ : Module NNRat G
    f g : ι → G
    ⊢ Eq (Fintype.balance (HSub.hSub f g)) (HSub.hSub (Fintype.balance f) (Fintype …
  -/
  simp only [balance, expect_sub_distrib, const_sub, sub_sub_sub_comm, Pi.sub_apply]
  /-
    🎉 no goals
  -/


@[simp] lemma balance_neg (f : ι → G) : balance (-f) = -balance f := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup G
    inst✝ : Module NNRat G
    f : ι → G
    ⊢ Eq (Fintype.balance (Neg.neg f)) (Neg.neg (Fintype.balance f))
  -/
  simp only [balance, expect_neg_distrib, const_neg, neg_sub', Pi.neg_apply]
  /-
    🎉 no goals
  -/


@[simp] lemma sum_balance (f : ι → G) : ∑ x, balance f x = 0 := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup G
    inst✝ : Module NNRat G
    f : ι → G
    ⊢ Eq (Finset.univ.sum fun x => Fintype.balance f x) 0
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases isEmpty_or_nonempty ι <;> simp [balance_apply]
                                  /-
                                    🎉 no goals
                                  -/


                                                                      /-
                                                                        ι : Type u_1
                                                                        G : Type u_4
                                                                        inst✝² : Fintype ι
                                                                        inst✝¹ : AddCommGroup G
                                                                        inst✝ : Module NNRat G
                                                                        f : ι → G
                                                                        ⊢ Eq (Finset.univ.expect fun x => Fintype.balance f x) 0
                                                                      -/
@[simp] lemma expect_balance (f : ι → G) : 𝔼 x, balance f x = 0 := by simp [expect]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp] lemma balance_idem (f : ι → G) : balance (balance f) = balance f := by
  /-
    ι : Type u_1
    G : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : AddCommGroup G
    inst✝ : Module NNRat G
    f : ι → G
    ⊢ Eq (Fintype.balance (Fintype.balance f)) (Fintype.balance f)
  -/
                                            /-
                                              🎉 no goals
                                            -/
  cases isEmpty_or_nonempty ι <;> ext x <;> simp [balance, expect_sub_distrib, univ_nonempty]
                                            /-
                                              🎉 no goals
                                            -/


@[simp] lemma map_balance [FunLike F G H] [LinearMapClass F ℚ≥0 G H] (g : F) (f : ι → G) (a : ι) :
                                              /-
                                                ι : Type u_1
                                                H : Type u_2
                                                F : Type u_3
                                                G : Type u_4
                                                inst✝⁶ : Fintype ι
                                                inst✝⁵ : AddCommGroup G
                                                inst✝⁴ : Module NNRat G
                                                inst✝³ : AddCommGroup H
                                                inst✝² : Module NNRat H
                                                inst✝¹ : FunLike F G H
                                                inst✝ : LinearMapClass F NNRat G H
                                                g : F
                                                f : ι → G
                                                a : ι
                                                ⊢ Eq (g (Fintype.balance f a)) (Fintype.balance (Function.comp (⇑g) f) a)
                                              -/
    g (balance f a) = balance (g ∘ f) a := by simp [balance, map_expect]
                                              /-
                                                🎉 no goals
                                              -/


