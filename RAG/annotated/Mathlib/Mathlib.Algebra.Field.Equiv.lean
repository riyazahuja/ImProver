protected theorem isField {A : Type*} (B : Type*) [Semiring A] [Semiring B] (hB : IsField B)
    (e : A ≃* B) : IsField A where
  exists_pair_ne := have ⟨x, y, h⟩ := hB.exists_pair_ne; ⟨e.symm x, e.symm y, e.symm.injective.ne h⟩
                                           /-
                                             A : Type u_1
                                             B : Type u_2
                                             inst✝¹ : Semiring A
                                             inst✝ : Semiring B
                                             hB : IsField B
                                             e : MulEquiv A B
                                             x y : A
                                             ⊢ Eq (e (HMul.hMul x y)) (e (HMul.hMul y x))
                                           -/
  mul_comm := fun x y => e.injective <| by rw [map_mul, map_mul, hB.mul_comm]
                                           /-
                                             🎉 no goals
                                           -/
  mul_inv_cancel := fun h => by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : Semiring A
      inst✝ : Semiring B
      hB : IsField B
      e : MulEquiv A B
      a✝ : A
      h : Ne a✝ 0
      ⊢ Exists fun b => Eq (HMul.hMul a✝ b) 1
    -/
    obtain ⟨a', he⟩ := hB.mul_inv_cancel ((e.injective.ne h).trans_eq <| map_zero e)
    /-
      case intro
      A : Type u_1
      B : Type u_2
      inst✝¹ : Semiring A
      inst✝ : Semiring B
      hB : IsField B
      e : MulEquiv A B
      a✝ : A
      h : Ne a✝ 0
      a' : B
      he : Eq (HMul.hMul (e a✝) a') 1
      ⊢ Exists fun b => Eq (HMul.hMul a✝ b) 1
    -/
    exact ⟨e.symm a', e.injective <| by rw [map_mul, map_one, e.apply_symm_apply, he]⟩
    /-
      🎉 no goals
    -/


