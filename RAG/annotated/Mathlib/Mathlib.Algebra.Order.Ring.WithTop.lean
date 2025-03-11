instance instMulZeroClass : MulZeroClass (WithTop α) where
  zero := 0
  mul a b := match a, b with
    | (a : α), (b : α) => ↑(a * b)
    | (a : α), ⊤ => if a = 0 then 0 else ⊤
    | ⊤, (b : α) => if b = 0 then 0 else ⊤
    | ⊤, ⊤ => ⊤
  mul_zero a := match a with
    | (a : α) => congr_arg some <| mul_zero _
    | ⊤ => if_pos rfl
  zero_mul b := match b with
    | (b : α) => congr_arg some <| zero_mul _
    | ⊤ => if_pos rfl


@[simp, norm_cast] lemma coe_mul (a b : α) : (↑(a * b) : WithTop α) = a * b := rfl


lemma mul_top' : ∀ (a : WithTop α), a * ⊤ = if a = 0 then 0 else ⊤
  | (a : α) => if_congr coe_eq_zero.symm rfl rfl
  | ⊤ => (if_neg top_ne_zero).symm


                                                    /-
                                                      α : Type u_1
                                                      inst✝¹ : DecidableEq α
                                                      inst✝ : MulZeroClass α
                                                      a : WithTop α
                                                      h : Ne a 0
                                                      ⊢ Eq (HMul.hMul a Top.top) Top.top
                                                    -/
@[simp] lemma mul_top (h : a ≠ 0) : a * ⊤ = ⊤ := by rw [mul_top', if_neg h]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma top_mul' : ∀ (b : WithTop α), ⊤ * b = if b = 0 then 0 else ⊤
  | (b : α) => if_congr coe_eq_zero.symm rfl rfl
  | ⊤ => (if_neg top_ne_zero).symm


                                                     /-
                                                       α : Type u_1
                                                       inst✝¹ : DecidableEq α
                                                       inst✝ : MulZeroClass α
                                                       b : WithTop α
                                                       hb : Ne b 0
                                                       ⊢ Eq (HMul.hMul Top.top b) Top.top
                                                     -/
@[simp] lemma top_mul (hb : b ≠ 0) : ⊤ * b = ⊤ := by rw [top_mul', if_neg hb]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma top_mul_top : (⊤ * ⊤ : WithTop α) = ⊤ := rfl


lemma mul_def (a b : WithTop α) :
    a * b = if a = 0 ∨ b = 0 then 0 else WithTop.map₂ (· * ·) a b := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithTop α
    ⊢ Eq (HMul.hMul a b) (ite (Or (Eq a 0) (Eq b 0)) 0 (WithTop.map₂ (fun x1 x2 => …
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases a <;> cases b <;> aesop
                          /-
                            🎉 no goals
                          -/


                                                                       /-
                                                                         α : Type u_1
                                                                         inst✝¹ : DecidableEq α
                                                                         inst✝ : MulZeroClass α
                                                                         a b : WithTop α
                                                                         ⊢ Iff (Eq (HMul.hMul a b) Top.top) (Or (And (Ne a 0) (Eq b Top.top)) (And (Eq  …
                                                                       -/
lemma mul_eq_top_iff : a * b = ⊤ ↔ a ≠ 0 ∧ b = ⊤ ∨ a = ⊤ ∧ b ≠ 0 := by rw [mul_def]; aesop
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma mul_coe_eq_bind {b : α} (hb : b ≠ 0) : ∀ a, (a * b : WithTop α) = a.bind fun a ↦ ↑(a * b)
            /-
              α : Type u_1
              inst✝¹ : DecidableEq α
              inst✝ : MulZeroClass α
              b : α
              hb : Ne b 0
              ⊢ Eq (HMul.hMul Top.top ↑b) (Option.bind Top.top fun a => Option.some (HMul.hM …
            -/
  | ⊤ => by simp [top_mul, hb]; rfl
                                /-
                                  🎉 no goals
                                -/
  | (a : α) => rfl


lemma coe_mul_eq_bind {a : α} (ha : a ≠ 0) : ∀ b, (a * b : WithTop α) = b.bind fun b ↦ ↑(a * b)
            /-
              α : Type u_1
              inst✝¹ : DecidableEq α
              inst✝ : MulZeroClass α
              a : α
              ha : Ne a 0
              ⊢ Eq (HMul.hMul (↑a) Top.top) (Option.bind Top.top fun b => Option.some (HMul. …
            -/
  | ⊤ => by simp [top_mul, ha]; rfl
                                /-
                                  🎉 no goals
                                -/
  | (b : α) => rfl


@[simp] lemma untop'_zero_mul (a b : WithTop α) : (a * b).untop' 0 = a.untop' 0 * b.untop' 0 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithTop α
    ⊢ Eq (WithTop.untop' 0 (HMul.hMul a b)) (HMul.hMul (WithTop.untop' 0 a) (WithT …
  -/
  by_cases ha : a = 0; · rw [ha, zero_mul, ← coe_zero, untop'_coe, zero_mul]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithTop α
    ha : Not (Eq a 0)
    ⊢ Eq (WithTop.untop' 0 (HMul.hMul a b)) (HMul.hMul (WithTop.untop' 0 a) (WithT …
  -/
  by_cases hb : b = 0; · rw [hb, mul_zero, ← coe_zero, untop'_coe, mul_zero]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithTop α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq (WithTop.untop' 0 (HMul.hMul a b)) (HMul.hMul (WithTop.untop' 0 a) (WithT …
  -/
  induction a; · rw [top_mul hb, untop'_top, zero_mul]
                 /-
                   🎉 no goals
                 -/
  /-
    case neg.coe
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    b : WithTop α
    hb : Not (Eq b 0)
    a✝ : α
    ha : Not (Eq (↑a✝) 0)
    ⊢ Eq (WithTop.untop' 0 (HMul.hMul (↑a✝) b)) (HMul.hMul (WithTop.untop' 0 ↑a✝)  …
  -/
  induction b; · rw [mul_top ha, untop'_top, mul_zero]
                 /-
                   🎉 no goals
                 -/
  /-
    case neg.coe.coe
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a✝¹ : α
    ha : Not (Eq (↑a✝¹) 0)
    a✝ : α
    hb : Not (Eq (↑a✝) 0)
    ⊢ Eq (WithTop.untop' 0 (HMul.hMul ↑a✝¹ ↑a✝)) (HMul.hMul (WithTop.untop' 0 ↑a✝¹ …
  -/
  rw [← coe_mul, untop'_coe, untop'_coe, untop'_coe]
  /-
    🎉 no goals
  -/


theorem mul_ne_top {a b : WithTop α} (ha : a ≠ ⊤) (hb : b ≠ ⊤) : a * b ≠ ⊤ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithTop α
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Ne (HMul.hMul a b) Top.top
  -/
  simp [mul_eq_top_iff, *]
  /-
    🎉 no goals
  -/


theorem mul_lt_top [LT α] {a b : WithTop α} (ha : a < ⊤) (hb : b < ⊤) : a * b < ⊤ := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    inst✝ : LT α
    a b : WithTop α
    ha : LT.lt a Top.top
    hb : LT.lt b Top.top
    ⊢ LT.lt (HMul.hMul a b) Top.top
  -/
  rw [WithTop.lt_top_iff_ne_top] at *
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    inst✝ : LT α
    a b : WithTop α
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Ne (HMul.hMul a b) Top.top
  -/
  exact mul_ne_top ha hb
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-25")] alias mul_lt_top' := mul_lt_top


instance instNoZeroDivisors [NoZeroDivisors α] : NoZeroDivisors (WithTop α) := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    a b : WithTop α
    inst✝ : NoZeroDivisors α
    ⊢ NoZeroDivisors (WithTop α)
  -/
  refine ⟨fun h₁ => Decidable.byContradiction fun h₂ => ?_⟩
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    a b : WithTop α
    inst✝ : NoZeroDivisors α
    a✝ b✝ : WithTop α
    h₁ : Eq (HMul.hMul a✝ b✝) 0
    h₂ : Not (Or (Eq a✝ 0) (Eq b✝ 0))
    ⊢ False
  -/
  rw [mul_def, if_neg h₂] at h₁
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    a b : WithTop α
    inst✝ : NoZeroDivisors α
    a✝ b✝ : WithTop α
    h₁ : Eq (WithTop.map₂ (fun x1 x2 => HMul.hMul x1 x2) a✝ b✝) 0
    h₂ : Not (Or (Eq a✝ 0) (Eq b✝ 0))
    ⊢ False
  -/
  rcases Option.mem_map₂_iff.1 h₁ with ⟨a, b, (rfl : _ = _), (rfl : _ = _), hab⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : MulZeroClass α
    a✝ b✝ : WithTop α
    inst✝ : NoZeroDivisors α
    a b : α
    hab : Eq (HMul.hMul a b) 0
    h₁ : Eq (WithTop.map₂ (fun x1 x2 => HMul.hMul x1 x2) (Option.some a) (Option.s …
    h₂ : Not (Or (Eq (Option.some a) 0) (Eq (Option.some b) 0))
    ⊢ False
  -/
  exact h₂ ((eq_zero_or_eq_zero_of_mul_eq_zero hab).imp (congr_arg some) (congr_arg some))
  /-
    🎉 no goals
  -/


/-- `Nontrivial α` is needed here as otherwise we have `1 * ⊤ = ⊤` but also `0 * ⊤ = 0`. -/
instance instMulZeroOneClass [MulZeroOneClass α] [Nontrivial α] : MulZeroOneClass (WithTop α) where
  __ := instMulZeroClass
  one_mul a := match a with
    | ⊤ => mul_top (mt coe_eq_coe.1 one_ne_zero)
                    /-
                      α : Type u_1
                      inst✝² : DecidableEq α
                      inst✝¹ : MulZeroOneClass α
                      inst✝ : Nontrivial α
                      a✝ : WithTop α
                      a : α
                      ⊢ Eq (HMul.hMul 1 ↑a) ↑a
                    -/
    | (a : α) => by rw [← coe_one, ← coe_mul, one_mul]
                    /-
                      🎉 no goals
                    -/
  mul_one a := match a with
    | ⊤ => top_mul (mt coe_eq_coe.1 one_ne_zero)
                    /-
                      α : Type u_1
                      inst✝² : DecidableEq α
                      inst✝¹ : MulZeroOneClass α
                      inst✝ : Nontrivial α
                      a✝ : WithTop α
                      a : α
                      ⊢ Eq (HMul.hMul (↑a) 1) ↑a
                    -/
    | (a : α) => by rw [← coe_one, ← coe_mul, mul_one]
                    /-
                      🎉 no goals
                    -/


/-- A version of `WithTop.map` for `MonoidWithZeroHom`s. -/
@[simps (config := .asFn)]
protected def _root_.MonoidWithZeroHom.withTopMap {R S : Type*} [MulZeroOneClass R] [DecidableEq R]
    [Nontrivial R] [MulZeroOneClass S] [DecidableEq S] [Nontrivial S] (f : R →*₀ S)
    (hf : Function.Injective f) : WithTop R →*₀ WithTop S :=
  { f.toZeroHom.withTopMap, f.toMonoidHom.toOneHom.withTopMap with
    toFun := WithTop.map f
    map_mul' := fun x y => by
      have : ∀ z, map f z = 0 ↔ z = 0 := fun z =>
        (Option.map_injective hf).eq_iff' f.toZeroHom.withTopMap.map_zero
      /-
        α : Type u_1
        inst✝⁶ : DecidableEq α
        R : Type u_2
        S : Type u_3
        inst✝⁵ : MulZeroOneClass R
        inst✝⁴ : DecidableEq R
        inst✝³ : Nontrivial R
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom R S
        hf : Function.Injective ⇑f
        x y : WithTop R
        this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
        ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul …
      -/
      rcases Decidable.eq_or_ne x 0 with (rfl | hx)
        /-
          case inl
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          y : WithTop R
          this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul 0 y)) (HMul …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        inst✝⁶ : DecidableEq α
        R : Type u_2
        S : Type u_3
        inst✝⁵ : MulZeroOneClass R
        inst✝⁴ : DecidableEq R
        inst✝³ : Nontrivial R
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom R S
        hf : Function.Injective ⇑f
        x y : WithTop R
        this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
        hx : Ne x 0
        ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul …
      -/
      rcases Decidable.eq_or_ne y 0 with (rfl | hy)
        /-
          case inr.inl
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          x : WithTop R
          this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          hx : Ne x 0
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x 0)) (HMul …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr.inr
        α : Type u_1
        inst✝⁶ : DecidableEq α
        R : Type u_2
        S : Type u_3
        inst✝⁵ : MulZeroOneClass R
        inst✝⁴ : DecidableEq R
        inst✝³ : Nontrivial R
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom R S
        hf : Function.Injective ⇑f
        x y : WithTop R
        this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul …
      -/
      induction' x with x
        /-
          case inr.inr.top
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          y : WithTop R
          this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          hy : Ne y 0
          hx : Ne Top.top 0
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul Top.top y)) …
        -/
      · simp [hy, this]
        /-
          🎉 no goals
        -/
      /-
        case inr.inr.coe
        α : Type u_1
        inst✝⁶ : DecidableEq α
        R : Type u_2
        S : Type u_3
        inst✝⁵ : MulZeroOneClass R
        inst✝⁴ : DecidableEq R
        inst✝³ : Nontrivial R
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom R S
        hf : Function.Injective ⇑f
        y : WithTop R
        this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
        hy : Ne y 0
        x : R
        hx : Ne (↑x) 0
        ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) y)) (H …
      -/
      induction' y with y
        /-
          case inr.inr.coe.top
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          x : R
          hx : Ne (↑x) 0
          hy : Ne Top.top 0
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) Top.to …
        -/
      · have : (f x : WithTop S) ≠ 0 := by simpa [hf.eq_iff' (map_zero f)] using hx
        /-
          case inr.inr.coe.top
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          this✝ : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          x : R
          hx : Ne (↑x) 0
          hy : Ne Top.top 0
          this : Ne (↑(f x)) 0
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) Top.to …
        -/
        simp [mul_top hx, mul_top this]
        /-
          🎉 no goals
        -/
      · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: `simp [← coe_mul]` times out
        /-
          case inr.inr.coe.coe
          α : Type u_1
          inst✝⁶ : DecidableEq α
          R : Type u_2
          S : Type u_3
          inst✝⁵ : MulZeroOneClass R
          inst✝⁴ : DecidableEq R
          inst✝³ : Nontrivial R
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom R S
          hf : Function.Injective ⇑f
          this : ∀ (z : WithTop R), Iff (Eq (WithTop.map (⇑f) z) 0) (Eq z 0)
          x : R
          hx : Ne (↑x) 0
          y : R
          hy : Ne (↑y) 0
          ⊢ Eq ({ toFun := WithTop.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul ↑x ↑y)) (HM …
        -/
        simp only [map_coe, ← coe_mul, map_mul] }
        /-
          🎉 no goals
        -/


instance instSemigroupWithZero [SemigroupWithZero α] [NoZeroDivisors α] :
    SemigroupWithZero (WithTop α) where
  __ := instMulZeroClass
  mul_assoc a b c := by
    /-
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      a b c : WithTop α
      ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
    -/
    rcases eq_or_ne a 0 with (rfl | ha); · simp only [zero_mul]
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case inr
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      a b c : WithTop α
      ha : Ne a 0
      ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
    -/
    rcases eq_or_ne b 0 with (rfl | hb); · simp only [zero_mul, mul_zero]
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case inr.inr
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      a b c : WithTop α
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
    -/
    rcases eq_or_ne c 0 with (rfl | hc); · simp only [mul_zero]
                                           /-
                                             🎉 no goals
                                           -/
  -- Porting note: below needed to be rewritten due to changed `simp` behaviour for `coe`
    /-
      case inr.inr.inr
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      a b c : WithTop α
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul b c))
    -/
    induction' a with a; · simp [hb, hc]
                           /-
                             🎉 no goals
                           -/
    /-
      case inr.inr.inr.coe
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      b c : WithTop α
      hb : Ne b 0
      hc : Ne c 0
      a : α
      ha : Ne (↑a) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (↑a) b) c) (HMul.hMul (↑a) (HMul.hMul b c))
    -/
    induction' b with b; · simp [mul_top ha, top_mul hc]
                           /-
                             🎉 no goals
                           -/
    /-
      case inr.inr.inr.coe.coe
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      c : WithTop α
      hc : Ne c 0
      a : α
      ha : Ne (↑a) 0
      b : α
      hb : Ne (↑b) 0
      ⊢ Eq (HMul.hMul (HMul.hMul ↑a ↑b) c) (HMul.hMul (↑a) (HMul.hMul (↑b) c))
    -/
    induction' c with c
      /-
        case inr.inr.inr.coe.coe.top
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : SemigroupWithZero α
        inst✝ : NoZeroDivisors α
        a : α
        ha : Ne (↑a) 0
        b : α
        hb : Ne (↑b) 0
        hc : Ne Top.top 0
        ⊢ Eq (HMul.hMul (HMul.hMul ↑a ↑b) Top.top) (HMul.hMul (↑a) (HMul.hMul (↑b) Top …
      -/
    · rw [mul_top hb, mul_top ha]
      /-
        case inr.inr.inr.coe.coe.top
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : SemigroupWithZero α
        inst✝ : NoZeroDivisors α
        a : α
        ha : Ne (↑a) 0
        b : α
        hb : Ne (↑b) 0
        hc : Ne Top.top 0
        ⊢ Eq (HMul.hMul (HMul.hMul ↑a ↑b) Top.top) Top.top
      -/
      rw [← coe_zero, ne_eq, coe_eq_coe] at ha hb
      /-
        case inr.inr.inr.coe.coe.top
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : SemigroupWithZero α
        inst✝ : NoZeroDivisors α
        a : α
        ha : Not (Eq a 0)
        b : α
        hb : Not (Eq b 0)
        hc : Ne Top.top 0
        ⊢ Eq (HMul.hMul (HMul.hMul ↑a ↑b) Top.top) Top.top
      -/
      simp [ha, hb]
      /-
        🎉 no goals
      -/
    /-
      case inr.inr.inr.coe.coe.coe
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : SemigroupWithZero α
      inst✝ : NoZeroDivisors α
      a : α
      ha : Ne (↑a) 0
      b : α
      hb : Ne (↑b) 0
      c : α
      hc : Ne (↑c) 0
      ⊢ Eq (HMul.hMul (HMul.hMul ↑a ↑b) ↑c) (HMul.hMul (↑a) (HMul.hMul ↑b ↑c))
    -/
    simp only [← coe_mul, mul_assoc]
    /-
      🎉 no goals
    -/


instance instMonoidWithZero : MonoidWithZero (WithTop α) where
  __ := instMulZeroOneClass
  __ := instSemigroupWithZero
  npow n a := match a, n with
    | (a : α), n => ↑(a ^ n)
    | ⊤, 0 => 1
    | ⊤, _n + 1 => ⊤
                    /-
                      α : Type u_1
                      inst✝³ : DecidableEq α
                      inst✝² : MonoidWithZero α
                      inst✝¹ : NoZeroDivisors α
                      inst✝ : Nontrivial α
                      a : WithTop α
                      ⊢ Eq ((fun n a => WithTop.instMonoidWithZero.match_1 (fun a n => WithTop α) a  …
                    -/
                                /-
                                  🎉 no goals
                                -/
  npow_zero a := by cases a <;> simp
                                /-
                                  🎉 no goals
                                -/
                      /-
                        α : Type u_1
                        inst✝³ : DecidableEq α
                        inst✝² : MonoidWithZero α
                        inst✝¹ : NoZeroDivisors α
                        inst✝ : Nontrivial α
                        n : Nat
                        a : WithTop α
                        ⊢ Eq ((fun n a => WithTop.instMonoidWithZero.match_1 (fun a n => WithTop α) a  …
                      -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  npow_succ n a := by cases n <;> cases a <;> simp [pow_succ]
                                              /-
                                                🎉 no goals
                                              -/


@[simp, norm_cast] lemma coe_pow (a : α) (n : ℕ) : (↑(a ^ n) : WithTop α) = a ^ n := rfl


theorem top_pow {n : ℕ} (n_pos : 0 < n) : (⊤ : WithTop α) ^ n = ⊤ :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝³ : DecidableEq α
                                                   inst✝² : MonoidWithZero α
                                                   inst✝¹ : NoZeroDivisors α
                                                   inst✝ : Nontrivial α
                                                   n : Nat
                                                   n_pos : LT.lt 0 n
                                                   m : Nat
                                                   x✝ : LE.le (Nat.succ 0) m
                                                   hm : Eq (HPow.hPow Top.top m) Top.top
                                                   ⊢ Eq (HPow.hPow Top.top (HAdd.hAdd m 1)) Top.top
                                                 -/
  Nat.le_induction (pow_one _) (fun m _ hm => by rw [pow_succ, hm, top_mul_top]) _
                                                 /-
                                                   🎉 no goals
                                                 -/
    (Nat.succ_le_of_lt n_pos)


instance instCommMonoidWithZero [CommMonoidWithZero α] [NoZeroDivisors α] [Nontrivial α] :
    CommMonoidWithZero (WithTop α) where
  __ := instMonoidWithZero
                     /-
                       α : Type u_1
                       inst✝³ : DecidableEq α
                       inst✝² : CommMonoidWithZero α
                       inst✝¹ : NoZeroDivisors α
                       inst✝ : Nontrivial α
                       a b : WithTop α
                       ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
                     -/
  mul_comm a b := by simp_rw [mul_def]; exact if_congr or_comm rfl (Option.map₂_comm mul_comm)
                                        /-
                                          🎉 no goals
                                        -/


private theorem distrib' (a b c : WithTop α) : (a + b) * c = a * c + b * c := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CanonicallyOrderedCommSemiring α
    a b c : WithTop α
    ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
  -/
  induction' c with c
    /-
      case top
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CanonicallyOrderedCommSemiring α
      a b : WithTop α
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) Top.top) (HAdd.hAdd (HMul.hMul a Top.top) (HMu …
    -/
                            /-
                              🎉 no goals
                            -/
  · by_cases ha : a = 0 <;> simp [ha]
                            /-
                              🎉 no goals
                            -/
    /-
      case coe
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CanonicallyOrderedCommSemiring α
      a b : WithTop α
      c : α
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑c) (HAdd.hAdd (HMul.hMul a ↑c) (HMul.hMul b ↑ …
    -/
  · by_cases hc : c = 0
      /-
        case pos
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : CanonicallyOrderedCommSemiring α
        a b : WithTop α
        c : α
        hc : Eq c 0
        ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑c) (HAdd.hAdd (HMul.hMul a ↑c) (HMul.hMul b ↑ …
      -/
    · simp [hc]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CanonicallyOrderedCommSemiring α
      a b : WithTop α
      c : α
      hc : Not (Eq c 0)
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ↑c) (HAdd.hAdd (HMul.hMul a ↑c) (HMul.hMul b ↑ …
    -/
    simp only [mul_coe_eq_bind hc]
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CanonicallyOrderedCommSemiring α
      a b : WithTop α
      c : α
      hc : Not (Eq c 0)
      ⊢ Eq (Option.bind (HAdd.hAdd a b) fun a => Option.some (HMul.hMul a c)) (HAdd. …
    -/
    cases a <;> cases b
    /-
      case neg.top.top
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : CanonicallyOrderedCommSemiring α
      c : α
      hc : Not (Eq c 0)
      ⊢ Eq (Option.bind (HAdd.hAdd Top.top Top.top) fun a => Option.some (HMul.hMul  …
    -/
    repeat' first | rfl |exact congr_arg some (add_mul _ _ _)
    /-
      🎉 no goals
    -/


/-- This instance requires `CanonicallyOrderedCommSemiring` as it is the smallest class
that derives from both `NonAssocNonUnitalSemiring` and `CanonicallyOrderedAddCommMonoid`, both
of which are required for distributivity. -/
instance commSemiring [Nontrivial α] : CommSemiring (WithTop α) :=
  { addCommMonoidWithOne, instCommMonoidWithZero with
    right_distrib := distrib'
    left_distrib := fun a b c => by
      /-
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : CanonicallyOrderedCommSemiring α
        inst✝ : Nontrivial α
        a b c : WithTop α
        ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
      -/
      rw [mul_comm, distrib', mul_comm b, mul_comm c] }
      /-
        🎉 no goals
      -/


instance [Nontrivial α] : CanonicallyOrderedCommSemiring (WithTop α) :=
  { WithTop.commSemiring, WithTop.canonicallyOrderedAddCommMonoid with
  eq_zero_or_eq_zero_of_mul_eq_zero := eq_zero_or_eq_zero_of_mul_eq_zero}


/-- A version of `WithTop.map` for `RingHom`s. -/
@[simps (config := .asFn)]
protected def _root_.RingHom.withTopMap {R S : Type*} [CanonicallyOrderedCommSemiring R]
    [DecidableEq R] [Nontrivial R] [CanonicallyOrderedCommSemiring S] [DecidableEq S] [Nontrivial S]
    (f : R →+* S) (hf : Function.Injective f) : WithTop R →+* WithTop S :=
  {MonoidWithZeroHom.withTopMap f.toMonoidWithZeroHom hf, f.toAddMonoidHom.withTopMap with}


@[gcongr]
protected lemma mul_lt_mul (ha : a₁ < a₂) (hb : b₁ < b₂) : a₁ * b₁ < a₂ * b₂ := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₁ a₂ b₁ b₂ : WithTop α
    ha : LT.lt a₁ a₂
    hb : LT.lt b₁ b₂
    ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
  -/
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹_›
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₁ a₂ b₁ b₂ : WithTop α
    ha : LT.lt a₁ a₂
    hb : LT.lt b₁ b₂
    this : MulPosStrictMono α
    ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
  -/
  lift a₁ to α using ha.lt_top.ne
  /-
    case intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₂ b₁ b₂ : WithTop α
    hb : LT.lt b₁ b₂
    this : MulPosStrictMono α
    a₁ : α
    ha : LT.lt (↑a₁) a₂
    ⊢ LT.lt (HMul.hMul (↑a₁) b₁) (HMul.hMul a₂ b₂)
  -/
  lift b₁ to α using hb.lt_top.ne
  /-
    case intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₂ b₂ : WithTop α
    this : MulPosStrictMono α
    a₁ : α
    ha : LT.lt (↑a₁) a₂
    b₁ : α
    hb : LT.lt (↑b₁) b₂
    ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul a₂ b₂)
  -/
  obtain rfl | ha₂ := eq_or_ne a₂ ⊤
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      b₂ : WithTop α
      this : MulPosStrictMono α
      a₁ b₁ : α
      hb : LT.lt (↑b₁) b₂
      ha : LT.lt (↑a₁) Top.top
      ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul Top.top b₂)
    -/
  · rw [top_mul (by simpa [bot_eq_zero] using hb.bot_lt.ne')]
    /-
      case intro.intro.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      b₂ : WithTop α
      this : MulPosStrictMono α
      a₁ b₁ : α
      hb : LT.lt (↑b₁) b₂
      ha : LT.lt (↑a₁) Top.top
      ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) Top.top
    -/
    exact coe_lt_top _
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₂ b₂ : WithTop α
    this : MulPosStrictMono α
    a₁ : α
    ha : LT.lt (↑a₁) a₂
    b₁ : α
    hb : LT.lt (↑b₁) b₂
    ha₂ : Ne a₂ Top.top
    ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul a₂ b₂)
  -/
  obtain rfl | hb₂ := eq_or_ne b₂ ⊤
    /-
      case intro.intro.inr.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      a₂ : WithTop α
      this : MulPosStrictMono α
      a₁ : α
      ha : LT.lt (↑a₁) a₂
      b₁ : α
      ha₂ : Ne a₂ Top.top
      hb : LT.lt (↑b₁) Top.top
      ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul a₂ Top.top)
    -/
  · rw [mul_top (by simpa [bot_eq_zero] using ha.bot_lt.ne')]
    /-
      case intro.intro.inr.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      a₂ : WithTop α
      this : MulPosStrictMono α
      a₁ : α
      ha : LT.lt (↑a₁) a₂
      b₁ : α
      ha₂ : Ne a₂ Top.top
      hb : LT.lt (↑b₁) Top.top
      ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) Top.top
    -/
    exact coe_lt_top _
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.inr
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    a₂ b₂ : WithTop α
    this : MulPosStrictMono α
    a₁ : α
    ha : LT.lt (↑a₁) a₂
    b₁ : α
    hb : LT.lt (↑b₁) b₂
    ha₂ : Ne a₂ Top.top
    hb₂ : Ne b₂ Top.top
    ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul a₂ b₂)
  -/
  lift a₂ to α using ha₂
  /-
    case intro.intro.inr.inr.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    b₂ : WithTop α
    this : MulPosStrictMono α
    a₁ b₁ : α
    hb : LT.lt (↑b₁) b₂
    hb₂ : Ne b₂ Top.top
    a₂ : α
    ha : LT.lt ↑a₁ ↑a₂
    ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul (↑a₂) b₂)
  -/
  lift b₂ to α using hb₂
  /-
    case intro.intro.inr.inr.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    this : MulPosStrictMono α
    a₁ b₁ a₂ : α
    ha : LT.lt ↑a₁ ↑a₂
    b₂ : α
    hb : LT.lt ↑b₁ ↑b₂
    ⊢ LT.lt (HMul.hMul ↑a₁ ↑b₁) (HMul.hMul ↑a₂ ↑b₂)
  -/
  norm_cast at *
  /-
    case intro.intro.inr.inr.intro.intro
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : CanonicallyOrderedCommSemiring α
    inst✝ : PosMulStrictMono α
    this : MulPosStrictMono α
    a₁ b₁ a₂ b₂ : α
    ha : LT.lt a₁ a₂
    hb : LT.lt b₁ b₂
    ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
  -/
  obtain rfl | hb₁ := eq_zero_or_pos b₁
    /-
      case intro.intro.inr.inr.intro.intro.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      this : MulPosStrictMono α
      a₁ a₂ b₂ : α
      ha : LT.lt a₁ a₂
      hb : LT.lt 0 b₂
      ⊢ LT.lt (HMul.hMul a₁ 0) (HMul.hMul a₂ b₂)
    -/
  · rw [mul_zero]
    /-
      case intro.intro.inr.inr.intro.intro.inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      this : MulPosStrictMono α
      a₁ a₂ b₂ : α
      ha : LT.lt a₁ a₂
      hb : LT.lt 0 b₂
      ⊢ LT.lt 0 (HMul.hMul a₂ b₂)
    -/
    exact mul_pos (by simpa [bot_eq_zero] using ha.bot_lt) hb
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr.intro.intro.inr
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : CanonicallyOrderedCommSemiring α
      inst✝ : PosMulStrictMono α
      this : MulPosStrictMono α
      a₁ b₁ a₂ b₂ : α
      ha : LT.lt a₁ a₂
      hb : LT.lt b₁ b₂
      hb₁ : LT.lt 0 b₁
      ⊢ LT.lt (HMul.hMul a₁ b₁) (HMul.hMul a₂ b₂)
    -/
  · exact mul_lt_mul ha hb.le hb₁ (zero_le _)
    /-
      🎉 no goals
    -/


protected lemma pow_right_strictMono : ∀ {n : ℕ}, n ≠ 0 → StrictMono fun a : WithTop α ↦ a ^ n
  | 0, h => absurd rfl h
               /-
                 α : Type u_1
                 inst✝⁴ : DecidableEq α
                 inst✝³ : CanonicallyOrderedCommSemiring α
                 inst✝² : PosMulStrictMono α
                 inst✝¹ : NoZeroDivisors α
                 inst✝ : Nontrivial α
                 x✝ : Ne 1 0
                 ⊢ StrictMono fun a => HPow.hPow a 1
               -/
  | 1, _ => by simpa only [pow_one] using strictMono_id
               /-
                 🎉 no goals
               -/
  | n + 2, _ => fun x y h ↦ by
    /-
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : PosMulStrictMono α
      inst✝¹ : NoZeroDivisors α
      inst✝ : Nontrivial α
      n : Nat
      x✝ : Ne (HAdd.hAdd n 2) 0
      x y : WithTop α
      h : LT.lt x y
      ⊢ LT.lt ((fun a => HPow.hPow a (HAdd.hAdd n 2)) x) ((fun a => HPow.hPow a (HAd …
    -/
    simp_rw [pow_succ _ (n + 1)]
    /-
      α : Type u_1
      inst✝⁴ : DecidableEq α
      inst✝³ : CanonicallyOrderedCommSemiring α
      inst✝² : PosMulStrictMono α
      inst✝¹ : NoZeroDivisors α
      inst✝ : Nontrivial α
      n : Nat
      x✝ : Ne (HAdd.hAdd n 2) 0
      x y : WithTop α
      h : LT.lt x y
      ⊢ LT.lt (HMul.hMul (HPow.hPow x (HAdd.hAdd n 1)) x) (HMul.hMul (HPow.hPow y (H …
    -/
    exact WithTop.mul_lt_mul (WithTop.pow_right_strictMono n.succ_ne_zero h) h
    /-
      🎉 no goals
    -/


@[gcongr] protected lemma pow_lt_pow_left (hab : a < b) {n : ℕ} (hn : n ≠ 0) : a ^ n < b ^ n :=
  WithTop.pow_right_strictMono hn hab


instance : MulZeroClass (WithBot α) := WithTop.instMulZeroClass


@[simp, norm_cast] lemma coe_mul (a b : α) : (↑(a * b) : WithBot α) = a * b := rfl


lemma mul_bot' : ∀ (a : WithBot α), a * ⊥ = if a = 0 then 0 else ⊥
  | (a : α) => if_congr coe_eq_zero.symm rfl rfl
  | ⊥ => (if_neg bot_ne_zero).symm


                                                    /-
                                                      α : Type u_1
                                                      inst✝¹ : DecidableEq α
                                                      inst✝ : MulZeroClass α
                                                      a : WithBot α
                                                      h : Ne a 0
                                                      ⊢ Eq (HMul.hMul a Bot.bot) Bot.bot
                                                    -/
@[simp] lemma mul_bot (h : a ≠ 0) : a * ⊥ = ⊥ := by rw [mul_bot', if_neg h]
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma bot_mul' : ∀ (b : WithBot α), ⊥ * b = if b = 0 then 0 else ⊥
  | (b : α) => if_congr coe_eq_zero.symm rfl rfl
  | ⊥ => (if_neg bot_ne_zero).symm


                                                     /-
                                                       α : Type u_1
                                                       inst✝¹ : DecidableEq α
                                                       inst✝ : MulZeroClass α
                                                       b : WithBot α
                                                       hb : Ne b 0
                                                       ⊢ Eq (HMul.hMul Bot.bot b) Bot.bot
                                                     -/
@[simp] lemma bot_mul (hb : b ≠ 0) : ⊥ * b = ⊥ := by rw [bot_mul', if_neg hb]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] lemma bot_mul_bot : (⊥ * ⊥ : WithBot α) = ⊥ := rfl


lemma mul_def (a b : WithBot α) :
    a * b = if a = 0 ∨ b = 0 then 0 else WithBot.map₂ (· * ·) a b := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithBot α
    ⊢ Eq (HMul.hMul a b) (ite (Or (Eq a 0) (Eq b 0)) 0 (WithBot.map₂ (fun x1 x2 => …
  -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
  cases a <;> cases b <;> aesop
                          /-
                            🎉 no goals
                          -/


                                                                       /-
                                                                         α : Type u_1
                                                                         inst✝¹ : DecidableEq α
                                                                         inst✝ : MulZeroClass α
                                                                         a b : WithBot α
                                                                         ⊢ Iff (Eq (HMul.hMul a b) Bot.bot) (Or (And (Ne a 0) (Eq b Bot.bot)) (And (Eq  …
                                                                       -/
lemma mul_eq_bot_iff : a * b = ⊥ ↔ a ≠ 0 ∧ b = ⊥ ∨ a = ⊥ ∧ b ≠ 0 := by rw [mul_def]; aesop
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma mul_coe_eq_bind {b : α} (hb : b ≠ 0) : ∀ a, (a * b : WithBot α) = a.bind fun a ↦ ↑(a * b)
            /-
              α : Type u_1
              inst✝¹ : DecidableEq α
              inst✝ : MulZeroClass α
              b : α
              hb : Ne b 0
              ⊢ Eq (HMul.hMul Bot.bot ↑b) (Option.bind Bot.bot fun a => Option.some (HMul.hM …
            -/
  | ⊥ => by simp only [ne_eq, coe_eq_zero, hb, not_false_eq_true, bot_mul]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  | (a : α) => rfl


lemma coe_mul_eq_bind {a : α} (ha : a ≠ 0) : ∀ b, (a * b : WithBot α) = b.bind fun b ↦ ↑(a * b)
            /-
              α : Type u_1
              inst✝¹ : DecidableEq α
              inst✝ : MulZeroClass α
              a : α
              ha : Ne a 0
              ⊢ Eq (HMul.hMul (↑a) Bot.bot) (Option.bind Bot.bot fun b => Option.some (HMul. …
            -/
  | ⊥ => by simp only [ne_eq, coe_eq_zero, ha, not_false_eq_true, mul_bot]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  | (b : α) => rfl


@[simp]
lemma unbot'_zero_mul (a b : WithBot α) : (a * b).unbot' 0 = a.unbot' 0 * b.unbot' 0 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithBot α
    ⊢ Eq (WithBot.unbot' 0 (HMul.hMul a b)) (HMul.hMul (WithBot.unbot' 0 a) (WithB …
  -/
  by_cases ha : a = 0; · rw [ha, zero_mul, ← coe_zero, unbot'_coe, zero_mul]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithBot α
    ha : Not (Eq a 0)
    ⊢ Eq (WithBot.unbot' 0 (HMul.hMul a b)) (HMul.hMul (WithBot.unbot' 0 a) (WithB …
  -/
  by_cases hb : b = 0; · rw [hb, mul_zero, ← coe_zero, unbot'_coe, mul_zero]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a b : WithBot α
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Eq (WithBot.unbot' 0 (HMul.hMul a b)) (HMul.hMul (WithBot.unbot' 0 a) (WithB …
  -/
  induction a; · rw [bot_mul hb, unbot'_bot, zero_mul]
                 /-
                   🎉 no goals
                 -/
  /-
    case neg.coe
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    b : WithBot α
    hb : Not (Eq b 0)
    a✝ : α
    ha : Not (Eq (↑a✝) 0)
    ⊢ Eq (WithBot.unbot' 0 (HMul.hMul (↑a✝) b)) (HMul.hMul (WithBot.unbot' 0 ↑a✝)  …
  -/
  induction b; · rw [mul_bot ha, unbot'_bot, mul_zero]
                 /-
                   🎉 no goals
                 -/
  /-
    case neg.coe.coe
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : MulZeroClass α
    a✝¹ : α
    ha : Not (Eq (↑a✝¹) 0)
    a✝ : α
    hb : Not (Eq (↑a✝) 0)
    ⊢ Eq (WithBot.unbot' 0 (HMul.hMul ↑a✝¹ ↑a✝)) (HMul.hMul (WithBot.unbot' 0 ↑a✝¹ …
  -/
  rw [← coe_mul, unbot'_coe, unbot'_coe, unbot'_coe]
  /-
    🎉 no goals
  -/


theorem mul_ne_bot {a b : WithBot α} (ha : a ≠ ⊥) (hb : b ≠ ⊥) : a * b ≠ ⊥ :=
  WithTop.mul_ne_top (α := αᵒᵈ) ha hb


theorem bot_lt_mul [LT α] {a b : WithBot α} (ha : ⊥ < a) (hb : ⊥ < b) : ⊥ < a * b :=
  WithTop.mul_lt_top (α := αᵒᵈ) ha hb


@[deprecated (since := "2024-08-25")] alias bot_lt_mul' := bot_lt_mul


instance instNoZeroDivisors [NoZeroDivisors α] : NoZeroDivisors (WithBot α) :=
  WithTop.instNoZeroDivisors


/-- `Nontrivial α` is needed here as otherwise we have `1 * ⊥ = ⊥` but also `= 0 * ⊥ = 0`. -/
instance instMulZeroOneClass [MulZeroOneClass α] [Nontrivial α] : MulZeroOneClass (WithBot α) :=
  WithTop.instMulZeroOneClass


instance instSemigroupWithZero [SemigroupWithZero α] [NoZeroDivisors α] :
    SemigroupWithZero (WithBot α) := WithTop.instSemigroupWithZero


instance instMonoidWithZero : MonoidWithZero (WithBot α) := WithTop.instMonoidWithZero


@[simp, norm_cast] lemma coe_pow (a : α) (n : ℕ) : (↑(a ^ n) : WithBot α) = a ^ n := rfl


instance commMonoidWithZero [CommMonoidWithZero α] [NoZeroDivisors α] [Nontrivial α] :
    CommMonoidWithZero (WithBot α) := WithTop.instCommMonoidWithZero


instance commSemiring [CanonicallyOrderedCommSemiring α] [Nontrivial α] :
    CommSemiring (WithBot α) :=
  WithTop.commSemiring


instance [MulZeroClass α] [Preorder α] [PosMulMono α] : PosMulMono (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      ⊢ Covariant (Subtype fun x => LE.le 0 x) (WithBot α) (fun x y => HMul.hMul (↑x …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      ⊢ LE.le ((fun x y => HMul.hMul (↑x) y) ⟨x, x0⟩ a) ((fun x y => HMul.hMul (↑x)  …
    -/
    simp only [Subtype.coe_mk]
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      ⊢ LE.le (HMul.hMul x a) (HMul.hMul x b)
    -/
    rcases eq_or_ne x 0 with rfl | x0'
      /-
        case inl
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulMono α
        a b : WithBot α
        h : LE.le a b
        x0 : LE.le 0 0
        ⊢ LE.le (HMul.hMul 0 a) (HMul.hMul 0 b)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      x0' : Ne x 0
      ⊢ LE.le (HMul.hMul x a) (HMul.hMul x b)
    -/
    lift x to α
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulMono α
        x : WithBot α
        x0 : LE.le 0 x
        a b : WithBot α
        h : LE.le a b
        x0' : Ne x 0
        ⊢ Ne x Bot.bot
      -/
    · rintro rfl
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulMono α
        a b : WithBot α
        h : LE.le a b
        x0 : LE.le 0 Bot.bot
        x0' : Ne Bot.bot 0
        ⊢ False
      -/
      exact (WithBot.bot_lt_coe (0 : α)).not_le x0
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      a b : WithBot α
      h : LE.le a b
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      ⊢ LE.le (HMul.hMul (↑x) a) (HMul.hMul (↑x) b)
    -/
    induction a
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulMono α
        b : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LE.le Bot.bot b
        ⊢ LE.le (HMul.hMul (↑x) Bot.bot) (HMul.hMul (↑x) b)
      -/
    · simp_rw [mul_bot x0', bot_le]
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      b : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝ : α
      h : LE.le (↑a✝) b
      ⊢ LE.le (HMul.hMul ↑x ↑a✝) (HMul.hMul (↑x) b)
    -/
    induction b
      /-
        case inr.intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulMono α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        a✝ : α
        h : LE.le (↑a✝) Bot.bot
        ⊢ LE.le (HMul.hMul ↑x ↑a✝) (HMul.hMul (↑x) Bot.bot)
      -/
    · exact absurd h (bot_lt_coe _).not_le
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le ↑a✝¹ ↑a✝
      ⊢ LE.le (HMul.hMul ↑x ↑a✝¹) (HMul.hMul ↑x ↑a✝)
    -/
    simp only [← coe_mul, coe_le_coe] at *
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le a✝¹ a✝
      ⊢ LE.le (HMul.hMul x a✝¹) (HMul.hMul x a✝)
    -/
    norm_cast at x0
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulMono α
      x : α
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le a✝¹ a✝
      x0 : LE.le 0 x
      ⊢ LE.le (HMul.hMul x a✝¹) (HMul.hMul x a✝)
    -/
    exact mul_le_mul_of_nonneg_left h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [MulPosMono α] : MulPosMono (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      ⊢ Covariant (Subtype fun x => LE.le 0 x) (WithBot α) (fun x y => HMul.hMul y ↑ …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      ⊢ LE.le ((fun x y => HMul.hMul y ↑x) ⟨x, x0⟩ a) ((fun x y => HMul.hMul y ↑x) ⟨ …
    -/
    simp only [Subtype.coe_mk]
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      ⊢ LE.le (HMul.hMul a x) (HMul.hMul b x)
    -/
    rcases eq_or_ne x 0 with rfl | x0'
      /-
        case inl
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosMono α
        a b : WithBot α
        h : LE.le a b
        x0 : LE.le 0 0
        ⊢ LE.le (HMul.hMul a 0) (HMul.hMul b 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LE.le a b
      x0' : Ne x 0
      ⊢ LE.le (HMul.hMul a x) (HMul.hMul b x)
    -/
    lift x to α
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosMono α
        x : WithBot α
        x0 : LE.le 0 x
        a b : WithBot α
        h : LE.le a b
        x0' : Ne x 0
        ⊢ Ne x Bot.bot
      -/
    · rintro rfl
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosMono α
        a b : WithBot α
        h : LE.le a b
        x0 : LE.le 0 Bot.bot
        x0' : Ne Bot.bot 0
        ⊢ False
      -/
      exact (WithBot.bot_lt_coe (0 : α)).not_le x0
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      a b : WithBot α
      h : LE.le a b
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      ⊢ LE.le (HMul.hMul a ↑x) (HMul.hMul b ↑x)
    -/
    induction a
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosMono α
        b : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LE.le Bot.bot b
        ⊢ LE.le (HMul.hMul Bot.bot ↑x) (HMul.hMul b ↑x)
      -/
    · simp_rw [bot_mul x0', bot_le]
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      b : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝ : α
      h : LE.le (↑a✝) b
      ⊢ LE.le (HMul.hMul ↑a✝ ↑x) (HMul.hMul b ↑x)
    -/
    induction b
      /-
        case inr.intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosMono α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        a✝ : α
        h : LE.le (↑a✝) Bot.bot
        ⊢ LE.le (HMul.hMul ↑a✝ ↑x) (HMul.hMul Bot.bot ↑x)
      -/
    · exact absurd h (bot_lt_coe _).not_le
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le ↑a✝¹ ↑a✝
      ⊢ LE.le (HMul.hMul ↑a✝¹ ↑x) (HMul.hMul ↑a✝ ↑x)
    -/
    simp only [← coe_mul, coe_le_coe] at *
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le a✝¹ a✝
      ⊢ LE.le (HMul.hMul a✝¹ x) (HMul.hMul a✝ x)
    -/
    norm_cast at x0
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosMono α
      x : α
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LE.le a✝¹ a✝
      x0 : LE.le 0 x
      ⊢ LE.le (HMul.hMul a✝¹ x) (HMul.hMul a✝ x)
    -/
    exact mul_le_mul_of_nonneg_right h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [PosMulStrictMono α] : PosMulStrictMono (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      ⊢ Covariant (Subtype fun x => LT.lt 0 x) (WithBot α) (fun x y => HMul.hMul (↑x …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LT.lt a b
      ⊢ LT.lt ((fun x y => HMul.hMul (↑x) y) ⟨x, x0⟩ a) ((fun x y => HMul.hMul (↑x)  …
    -/
    simp only [Subtype.coe_mk]
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LT.lt a b
      ⊢ LT.lt (HMul.hMul x a) (HMul.hMul x b)
    -/
    lift x to α using x0.ne_bot
    /-
      case intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      a b : WithBot α
      h : LT.lt a b
      x : α
      x0 : LT.lt 0 ↑x
      ⊢ LT.lt (HMul.hMul (↑x) a) (HMul.hMul (↑x) b)
    -/
    induction b
      /-
        case intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulStrictMono α
        a : WithBot α
        x : α
        x0 : LT.lt 0 ↑x
        h : LT.lt a Bot.bot
        ⊢ LT.lt (HMul.hMul (↑x) a) (HMul.hMul (↑x) Bot.bot)
      -/
    · exact absurd h not_lt_bot
      /-
        🎉 no goals
      -/
    /-
      case intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      a : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      a✝ : α
      h : LT.lt a ↑a✝
      ⊢ LT.lt (HMul.hMul (↑x) a) (HMul.hMul ↑x ↑a✝)
    -/
    induction a
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulStrictMono α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LT.lt Bot.bot ↑a✝
        ⊢ LT.lt (HMul.hMul (↑x) Bot.bot) (HMul.hMul ↑x ↑a✝)
      -/
    · simp_rw [mul_bot x0.ne.symm, ← coe_mul, bot_lt_coe]
      /-
        🎉 no goals
      -/
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LT.lt ↑a✝ ↑a✝¹
      ⊢ LT.lt (HMul.hMul ↑x ↑a✝) (HMul.hMul ↑x ↑a✝¹)
    -/
    simp only [← coe_mul, coe_lt_coe] at *
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      ⊢ LT.lt (HMul.hMul x a✝) (HMul.hMul x a✝¹)
    -/
    norm_cast at x0
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulStrictMono α
      x a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      x0 : LT.lt 0 x
      ⊢ LT.lt (HMul.hMul x a✝) (HMul.hMul x a✝¹)
    -/
    exact mul_lt_mul_of_pos_left h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [MulPosStrictMono α] : MulPosStrictMono (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      ⊢ Covariant (Subtype fun x => LT.lt 0 x) (WithBot α) (fun x y => HMul.hMul y ↑ …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LT.lt a b
      ⊢ LT.lt ((fun x y => HMul.hMul y ↑x) ⟨x, x0⟩ a) ((fun x y => HMul.hMul y ↑x) ⟨ …
    -/
    simp only [Subtype.coe_mk]
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LT.lt a b
      ⊢ LT.lt (HMul.hMul a x) (HMul.hMul b x)
    -/
    lift x to α using x0.ne_bot
    /-
      case intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      a b : WithBot α
      h : LT.lt a b
      x : α
      x0 : LT.lt 0 ↑x
      ⊢ LT.lt (HMul.hMul a ↑x) (HMul.hMul b ↑x)
    -/
    induction b
      /-
        case intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosStrictMono α
        a : WithBot α
        x : α
        x0 : LT.lt 0 ↑x
        h : LT.lt a Bot.bot
        ⊢ LT.lt (HMul.hMul a ↑x) (HMul.hMul Bot.bot ↑x)
      -/
    · exact absurd h not_lt_bot
      /-
        🎉 no goals
      -/
    /-
      case intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      a : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      a✝ : α
      h : LT.lt a ↑a✝
      ⊢ LT.lt (HMul.hMul a ↑x) (HMul.hMul ↑a✝ ↑x)
    -/
    induction a
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosStrictMono α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LT.lt Bot.bot ↑a✝
        ⊢ LT.lt (HMul.hMul Bot.bot ↑x) (HMul.hMul ↑a✝ ↑x)
      -/
    · simp_rw [bot_mul x0.ne.symm, ← coe_mul, bot_lt_coe]
      /-
        🎉 no goals
      -/
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LT.lt ↑a✝ ↑a✝¹
      ⊢ LT.lt (HMul.hMul ↑a✝ ↑x) (HMul.hMul ↑a✝¹ ↑x)
    -/
    simp only [← coe_mul, coe_lt_coe] at *
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      ⊢ LT.lt (HMul.hMul a✝ x) (HMul.hMul a✝¹ x)
    -/
    norm_cast at x0
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosStrictMono α
      x a✝¹ a✝ : α
      h : LT.lt a✝ a✝¹
      x0 : LT.lt 0 x
      ⊢ LT.lt (HMul.hMul a✝ x) (HMul.hMul a✝¹ x)
    -/
    exact mul_lt_mul_of_pos_right h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [PosMulReflectLT α] : PosMulReflectLT (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      ⊢ Contravariant (Subtype fun x => LE.le 0 x) (WithBot α) (fun x y => HMul.hMul …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt ((fun x y => HMul.hMul (↑x) y) ⟨x, x0⟩ a) ((fun x y => HMul.hMul (↑x …
      ⊢ LT.lt a b
    -/
    simp only [Subtype.coe_mk] at h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt (HMul.hMul x a) (HMul.hMul x b)
      ⊢ LT.lt a b
    -/
    rcases eq_or_ne x 0 with rfl | x0'
      /-
        case inl
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        a b : WithBot α
        x0 : LE.le 0 0
        h : LT.lt (HMul.hMul 0 a) (HMul.hMul 0 b)
        ⊢ LT.lt a b
      -/
    · simp at h
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt (HMul.hMul x a) (HMul.hMul x b)
      x0' : Ne x 0
      ⊢ LT.lt a b
    -/
    lift x to α
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        x : WithBot α
        x0 : LE.le 0 x
        a b : WithBot α
        h : LT.lt (HMul.hMul x a) (HMul.hMul x b)
        x0' : Ne x 0
        ⊢ Ne x Bot.bot
      -/
    · rintro rfl
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        a b : WithBot α
        x0 : LE.le 0 Bot.bot
        h : LT.lt (HMul.hMul Bot.bot a) (HMul.hMul Bot.bot b)
        x0' : Ne Bot.bot 0
        ⊢ False
      -/
      exact (WithBot.bot_lt_coe (0 : α)).not_le x0
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      a b : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      h : LT.lt (HMul.hMul (↑x) a) (HMul.hMul (↑x) b)
      x0' : Ne (↑x) 0
      ⊢ LT.lt a b
    -/
    induction b
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        a : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LT.lt (HMul.hMul (↑x) a) (HMul.hMul (↑x) Bot.bot)
        ⊢ LT.lt a Bot.bot
      -/
    · rw [mul_bot x0'] at h
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        a : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LT.lt (HMul.hMul (↑x) a) Bot.bot
        ⊢ LT.lt a Bot.bot
      -/
      exact absurd h bot_le.not_lt
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      a : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝ : α
      h : LT.lt (HMul.hMul (↑x) a) (HMul.hMul ↑x ↑a✝)
      ⊢ LT.lt a ↑a✝
    -/
    induction a
      /-
        case inr.intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLT α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        a✝ : α
        h : LT.lt (HMul.hMul (↑x) Bot.bot) (HMul.hMul ↑x ↑a✝)
        ⊢ LT.lt Bot.bot ↑a✝
      -/
    · exact WithBot.bot_lt_coe _
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul ↑x ↑a✝) (HMul.hMul ↑x ↑a✝¹)
      ⊢ LT.lt ↑a✝ ↑a✝¹
    -/
    simp only [← coe_mul, coe_lt_coe] at *
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul x a✝) (HMul.hMul x a✝¹)
      ⊢ LT.lt a✝ a✝¹
    -/
    norm_cast at x0
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLT α
      x : α
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul x a✝) (HMul.hMul x a✝¹)
      x0 : LE.le 0 x
      ⊢ LT.lt a✝ a✝¹
    -/
    exact lt_of_mul_lt_mul_left h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [MulPosReflectLT α] : MulPosReflectLT (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      ⊢ Contravariant (Subtype fun x => LE.le 0 x) (WithBot α) (fun x y => HMul.hMul …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt ((fun x y => HMul.hMul y ↑x) ⟨x, x0⟩ a) ((fun x y => HMul.hMul y ↑x) …
      ⊢ LT.lt a b
    -/
    simp only [Subtype.coe_mk] at h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt (HMul.hMul a x) (HMul.hMul b x)
      ⊢ LT.lt a b
    -/
    rcases eq_or_ne x 0 with rfl | x0'
      /-
        case inl
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        a b : WithBot α
        x0 : LE.le 0 0
        h : LT.lt (HMul.hMul a 0) (HMul.hMul b 0)
        ⊢ LT.lt a b
      -/
    · simp at h
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : WithBot α
      x0 : LE.le 0 x
      a b : WithBot α
      h : LT.lt (HMul.hMul a x) (HMul.hMul b x)
      x0' : Ne x 0
      ⊢ LT.lt a b
    -/
    lift x to α
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        x : WithBot α
        x0 : LE.le 0 x
        a b : WithBot α
        h : LT.lt (HMul.hMul a x) (HMul.hMul b x)
        x0' : Ne x 0
        ⊢ Ne x Bot.bot
      -/
    · rintro rfl
      /-
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        a b : WithBot α
        x0 : LE.le 0 Bot.bot
        h : LT.lt (HMul.hMul a Bot.bot) (HMul.hMul b Bot.bot)
        x0' : Ne Bot.bot 0
        ⊢ False
      -/
      exact (WithBot.bot_lt_coe (0 : α)).not_le x0
      /-
        🎉 no goals
      -/
    /-
      case inr.intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      a b : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      h : LT.lt (HMul.hMul a ↑x) (HMul.hMul b ↑x)
      x0' : Ne (↑x) 0
      ⊢ LT.lt a b
    -/
    induction b
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        a : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LT.lt (HMul.hMul a ↑x) (HMul.hMul Bot.bot ↑x)
        ⊢ LT.lt a Bot.bot
      -/
    · rw [bot_mul x0'] at h
      /-
        case inr.intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        a : WithBot α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        h : LT.lt (HMul.hMul a ↑x) Bot.bot
        ⊢ LT.lt a Bot.bot
      -/
      exact absurd h bot_le.not_lt
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      a : WithBot α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝ : α
      h : LT.lt (HMul.hMul a ↑x) (HMul.hMul ↑a✝ ↑x)
      ⊢ LT.lt a ↑a✝
    -/
    induction a
      /-
        case inr.intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLT α
        x : α
        x0 : LE.le 0 ↑x
        x0' : Ne (↑x) 0
        a✝ : α
        h : LT.lt (HMul.hMul Bot.bot ↑x) (HMul.hMul ↑a✝ ↑x)
        ⊢ LT.lt Bot.bot ↑a✝
      -/
    · exact WithBot.bot_lt_coe _
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul ↑a✝ ↑x) (HMul.hMul ↑a✝¹ ↑x)
      ⊢ LT.lt ↑a✝ ↑a✝¹
    -/
    simp only [← coe_mul, coe_lt_coe] at *
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : α
      x0 : LE.le 0 ↑x
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul a✝ x) (HMul.hMul a✝¹ x)
      ⊢ LT.lt a✝ a✝¹
    -/
    norm_cast at x0
    /-
      case inr.intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLT α
      x : α
      x0' : Ne (↑x) 0
      a✝¹ a✝ : α
      h : LT.lt (HMul.hMul a✝ x) (HMul.hMul a✝¹ x)
      x0 : LE.le 0 x
      ⊢ LT.lt a✝ a✝¹
    -/
    exact lt_of_mul_lt_mul_right h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [PosMulReflectLE α] : PosMulReflectLE (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      ⊢ Contravariant (Subtype fun x => LT.lt 0 x) (WithBot α) (fun x y => HMul.hMul …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LE.le ((fun x y => HMul.hMul (↑x) y) ⟨x, x0⟩ a) ((fun x y => HMul.hMul (↑x …
      ⊢ LE.le a b
    -/
    simp only [Subtype.coe_mk] at h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LE.le (HMul.hMul x a) (HMul.hMul x b)
      ⊢ LE.le a b
    -/
    lift x to α using x0.ne_bot
    /-
      case intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      a b : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      h : LE.le (HMul.hMul (↑x) a) (HMul.hMul (↑x) b)
      ⊢ LE.le a b
    -/
    induction a
      /-
        case intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLE α
        b : WithBot α
        x : α
        x0 : LT.lt 0 ↑x
        h : LE.le (HMul.hMul (↑x) Bot.bot) (HMul.hMul (↑x) b)
        ⊢ LE.le Bot.bot b
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
    /-
      case intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      b : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      a✝ : α
      h : LE.le (HMul.hMul ↑x ↑a✝) (HMul.hMul (↑x) b)
      ⊢ LE.le (↑a✝) b
    -/
    induction b
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLE α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LE.le (HMul.hMul ↑x ↑a✝) (HMul.hMul (↑x) Bot.bot)
        ⊢ LE.le (↑a✝) Bot.bot
      -/
    · rw [mul_bot x0.ne.symm, ← coe_mul] at h
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : PosMulReflectLE α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LE.le (↑(HMul.hMul x a✝)) Bot.bot
        ⊢ LE.le (↑a✝) Bot.bot
      -/
      exact absurd h (bot_lt_coe _).not_le
      /-
        🎉 no goals
      -/
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LE.le (HMul.hMul ↑x ↑a✝¹) (HMul.hMul ↑x ↑a✝)
      ⊢ LE.le ↑a✝¹ ↑a✝
    -/
    simp only [← coe_mul, coe_le_coe] at *
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LE.le (HMul.hMul x a✝¹) (HMul.hMul x a✝)
      ⊢ LE.le a✝¹ a✝
    -/
    norm_cast at x0
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : PosMulReflectLE α
      x a✝¹ a✝ : α
      h : LE.le (HMul.hMul x a✝¹) (HMul.hMul x a✝)
      x0 : LT.lt 0 x
      ⊢ LE.le a✝¹ a✝
    -/
    exact le_of_mul_le_mul_left h x0 ⟩
    /-
      🎉 no goals
    -/


instance [MulZeroClass α] [Preorder α] [MulPosReflectLE α] : MulPosReflectLE (WithBot α) :=
  ⟨by
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      ⊢ Contravariant (Subtype fun x => LT.lt 0 x) (WithBot α) (fun x y => HMul.hMul …
    -/
    intro ⟨x, x0⟩ a b h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LE.le ((fun x y => HMul.hMul y ↑x) ⟨x, x0⟩ a) ((fun x y => HMul.hMul y ↑x) …
      ⊢ LE.le a b
    -/
    simp only [Subtype.coe_mk] at h
    /-
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      x : WithBot α
      x0 : LT.lt 0 x
      a b : WithBot α
      h : LE.le (HMul.hMul a x) (HMul.hMul b x)
      ⊢ LE.le a b
    -/
    lift x to α using x0.ne_bot
    /-
      case intro
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      a b : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      h : LE.le (HMul.hMul a ↑x) (HMul.hMul b ↑x)
      ⊢ LE.le a b
    -/
    induction a
      /-
        case intro.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLE α
        b : WithBot α
        x : α
        x0 : LT.lt 0 ↑x
        h : LE.le (HMul.hMul Bot.bot ↑x) (HMul.hMul b ↑x)
        ⊢ LE.le Bot.bot b
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
    /-
      case intro.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      b : WithBot α
      x : α
      x0 : LT.lt 0 ↑x
      a✝ : α
      h : LE.le (HMul.hMul ↑a✝ ↑x) (HMul.hMul b ↑x)
      ⊢ LE.le (↑a✝) b
    -/
    induction b
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLE α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LE.le (HMul.hMul ↑a✝ ↑x) (HMul.hMul Bot.bot ↑x)
        ⊢ LE.le (↑a✝) Bot.bot
      -/
    · rw [bot_mul x0.ne.symm, ← coe_mul] at h
      /-
        case intro.coe.bot
        α : Type u_1
        inst✝³ : DecidableEq α
        inst✝² : MulZeroClass α
        inst✝¹ : Preorder α
        inst✝ : MulPosReflectLE α
        x : α
        x0 : LT.lt 0 ↑x
        a✝ : α
        h : LE.le (↑(HMul.hMul a✝ x)) Bot.bot
        ⊢ LE.le (↑a✝) Bot.bot
      -/
      exact absurd h (bot_lt_coe _).not_le
      /-
        🎉 no goals
      -/
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LE.le (HMul.hMul ↑a✝¹ ↑x) (HMul.hMul ↑a✝ ↑x)
      ⊢ LE.le ↑a✝¹ ↑a✝
    -/
    simp only [← coe_mul, coe_le_coe] at *
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      x : α
      x0 : LT.lt 0 ↑x
      a✝¹ a✝ : α
      h : LE.le (HMul.hMul a✝¹ x) (HMul.hMul a✝ x)
      ⊢ LE.le a✝¹ a✝
    -/
    norm_cast at x0
    /-
      case intro.coe.coe
      α : Type u_1
      inst✝³ : DecidableEq α
      inst✝² : MulZeroClass α
      inst✝¹ : Preorder α
      inst✝ : MulPosReflectLE α
      x a✝¹ a✝ : α
      h : LE.le (HMul.hMul a✝¹ x) (HMul.hMul a✝ x)
      x0 : LT.lt 0 x
      ⊢ LE.le a✝¹ a✝
    -/
    exact le_of_mul_le_mul_right h x0 ⟩
    /-
      🎉 no goals
    -/


instance orderedCommSemiring [CanonicallyOrderedCommSemiring α] [Nontrivial α] :
    OrderedCommSemiring (WithBot α) :=
  { WithBot.zeroLEOneClass, WithBot.orderedAddCommMonoid, WithBot.commSemiring with
    mul_le_mul_of_nonneg_left  := fun _ _ _ => mul_le_mul_of_nonneg_left
    mul_le_mul_of_nonneg_right := fun _ _ _ => mul_le_mul_of_nonneg_right }


