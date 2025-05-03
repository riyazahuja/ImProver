/-- Given two multilinear maps `(ι₁ → N) → N₁` and `(ι₂ → N) → N₂`, this produces the map
`(ι₁ ⊕ ι₂ → N) → N₁ ⊗ N₂` by taking the coproduct of the domain and the tensor product
of the codomain.

This can be thought of as combining `Equiv.sumArrowEquivProdArrow.symm` with
`TensorProduct.map`, noting that the two operations can't be separated as the intermediate result
is not a `MultilinearMap`.

While this can be generalized to work for dependent `Π i : ι₁, N'₁ i` instead of `ι₁ → N`, doing so
introduces `Sum.elim N'₁ N'₂` types in the result which are difficult to work with and not defeq
to the simple case defined here. See [this zulip thread](
https://leanprover.zulipchat.com/#narrow/stream/217875-Is-there.20code.20for.20X.3F/topic/Instances.20on.20.60sum.2Eelim.20A.20B.20i.60/near/218484619).
-/
@[simps apply]
def domCoprod (a : MultilinearMap R (fun _ : ι₁ => N) N₁)
    (b : MultilinearMap R (fun _ : ι₂ => N) N₂) :
    MultilinearMap R (fun _ : ι₁ ⊕ ι₂ => N) (N₁ ⊗[R] N₂) where
  toFun v := (a fun i => v (Sum.inl i)) ⊗ₜ b fun i => v (Sum.inr i)
  map_update_add' _ i p q := by
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      p q : N
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
    letI := (@Sum.inl_injective ι₁ ι₂).decidableEq
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      p q : N
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
    letI := (@Sum.inr_injective ι₁ ι₂).decidableEq
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      p q : N
      this✝ : DecidableEq ι₁ := ⋯.decidableEq
      this : DecidableEq ι₂ := ⋯.decidableEq
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
                /-
                  🎉 no goals
                -/
    cases i <;> simp [TensorProduct.add_tmul, TensorProduct.tmul_add]
                /-
                  🎉 no goals
                -/
  map_update_smul' _ i c p := by
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      c : R
      p : N
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
    letI := (@Sum.inl_injective ι₁ ι₂).decidableEq
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      c : R
      p : N
      this : DecidableEq ι₁ := ⋯.decidableEq
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
    letI := (@Sum.inr_injective ι₁ ι₂).decidableEq
    /-
      R : Type u_1
      ι₁ : Type u_2
      ι₂ : Type u_3
      ι₃ : Type u_4
      ι₄ : Type u_5
      inst✝⁷ : CommSemiring R
      N₁ : Type u_6
      inst✝⁶ : AddCommMonoid N₁
      inst✝⁵ : Module R N₁
      N₂ : Type u_7
      inst✝⁴ : AddCommMonoid N₂
      inst✝³ : Module R N₂
      N : Type u_8
      inst✝² : AddCommMonoid N
      inst✝¹ : Module R N
      a : MultilinearMap R (fun x => N) N₁
      b : MultilinearMap R (fun x => N) N₂
      inst✝ : DecidableEq (Sum ι₁ ι₂)
      x✝ : Sum ι₁ ι₂ → N
      i : Sum ι₁ ι₂
      c : R
      p : N
      this✝ : DecidableEq ι₁ := ⋯.decidableEq
      this : DecidableEq ι₂ := ⋯.decidableEq
      ⊢ Eq ((fun v => TensorProduct.tmul R (a fun i => v (Sum.inl i)) (b fun i => v  …
    -/
                /-
                  🎉 no goals
                -/
    cases i <;> simp [TensorProduct.smul_tmul', TensorProduct.tmul_smul]
                /-
                  🎉 no goals
                -/


/-- A more bundled version of `MultilinearMap.domCoprod` that maps
`((ι₁ → N) → N₁) ⊗ ((ι₂ → N) → N₂)` to `(ι₁ ⊕ ι₂ → N) → N₁ ⊗ N₂`. -/
def domCoprod' :
    MultilinearMap R (fun _ : ι₁ => N) N₁ ⊗[R] MultilinearMap R (fun _ : ι₂ => N) N₂ →ₗ[R]
      MultilinearMap R (fun _ : ι₁ ⊕ ι₂ => N) (N₁ ⊗[R] N₂) :=
  TensorProduct.lift <|
    LinearMap.mk₂ R domCoprod
      (fun m₁ m₂ n => by
        /-
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          m₁ m₂ : MultilinearMap R (fun x => N) N₁
          n : MultilinearMap R (fun x => N) N₂
          ⊢ Eq ((HAdd.hAdd m₁ m₂).domCoprod n) (HAdd.hAdd (m₁.domCoprod n) (m₂.domCoprod …
        -/
        ext
        /-
          case H
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          m₁ m₂ : MultilinearMap R (fun x => N) N₁
          n : MultilinearMap R (fun x => N) N₂
          x✝ : Sum ι₁ ι₂ → N
          ⊢ Eq (((HAdd.hAdd m₁ m₂).domCoprod n) x✝) ((HAdd.hAdd (m₁.domCoprod n) (m₂.dom …
        -/
        simp only [domCoprod_apply, TensorProduct.add_tmul, add_apply])
        /-
          🎉 no goals
        -/
      (fun c m n => by
        /-
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          c : R
          m : MultilinearMap R (fun x => N) N₁
          n : MultilinearMap R (fun x => N) N₂
          ⊢ Eq ((HSMul.hSMul c m).domCoprod n) (HSMul.hSMul c (m.domCoprod n))
        -/
        ext
        /-
          case H
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          c : R
          m : MultilinearMap R (fun x => N) N₁
          n : MultilinearMap R (fun x => N) N₂
          x✝ : Sum ι₁ ι₂ → N
          ⊢ Eq (((HSMul.hSMul c m).domCoprod n) x✝) ((HSMul.hSMul c (m.domCoprod n)) x✝)
        -/
        simp only [domCoprod_apply, TensorProduct.smul_tmul', smul_apply])
        /-
          🎉 no goals
        -/
      (fun m n₁ n₂ => by
        /-
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          m : MultilinearMap R (fun x => N) N₁
          n₁ n₂ : MultilinearMap R (fun x => N) N₂
          ⊢ Eq (m.domCoprod (HAdd.hAdd n₁ n₂)) (HAdd.hAdd (m.domCoprod n₁) (m.domCoprod  …
        -/
        ext
        /-
          case H
          R : Type u_1
          ι₁ : Type u_2
          ι₂ : Type u_3
          ι₃ : Type u_4
          ι₄ : Type u_5
          inst✝⁶ : CommSemiring R
          N₁ : Type u_6
          inst✝⁵ : AddCommMonoid N₁
          inst✝⁴ : Module R N₁
          N₂ : Type u_7
          inst✝³ : AddCommMonoid N₂
          inst✝² : Module R N₂
          N : Type u_8
          inst✝¹ : AddCommMonoid N
          inst✝ : Module R N
          m : MultilinearMap R (fun x => N) N₁
          n₁ n₂ : MultilinearMap R (fun x => N) N₂
          x✝ : Sum ι₁ ι₂ → N
          ⊢ Eq ((m.domCoprod (HAdd.hAdd n₁ n₂)) x✝) ((HAdd.hAdd (m.domCoprod n₁) (m.domC …
        -/
        simp only [domCoprod_apply, TensorProduct.tmul_add, add_apply])
        /-
          🎉 no goals
        -/
      fun c m n => by
      /-
        R : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        ι₃ : Type u_4
        ι₄ : Type u_5
        inst✝⁶ : CommSemiring R
        N₁ : Type u_6
        inst✝⁵ : AddCommMonoid N₁
        inst✝⁴ : Module R N₁
        N₂ : Type u_7
        inst✝³ : AddCommMonoid N₂
        inst✝² : Module R N₂
        N : Type u_8
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        c : R
        m : MultilinearMap R (fun x => N) N₁
        n : MultilinearMap R (fun x => N) N₂
        ⊢ Eq (m.domCoprod (HSMul.hSMul c n)) (HSMul.hSMul c (m.domCoprod n))
      -/
      ext
      /-
        case H
        R : Type u_1
        ι₁ : Type u_2
        ι₂ : Type u_3
        ι₃ : Type u_4
        ι₄ : Type u_5
        inst✝⁶ : CommSemiring R
        N₁ : Type u_6
        inst✝⁵ : AddCommMonoid N₁
        inst✝⁴ : Module R N₁
        N₂ : Type u_7
        inst✝³ : AddCommMonoid N₂
        inst✝² : Module R N₂
        N : Type u_8
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        c : R
        m : MultilinearMap R (fun x => N) N₁
        n : MultilinearMap R (fun x => N) N₂
        x✝ : Sum ι₁ ι₂ → N
        ⊢ Eq ((m.domCoprod (HSMul.hSMul c n)) x✝) ((HSMul.hSMul c (m.domCoprod n)) x✝)
      -/
      simp only [domCoprod_apply, TensorProduct.tmul_smul, smul_apply]
      /-
        🎉 no goals
      -/


@[simp]
theorem domCoprod'_apply (a : MultilinearMap R (fun _ : ι₁ => N) N₁)
    (b : MultilinearMap R (fun _ : ι₂ => N) N₂) : domCoprod' (a ⊗ₜ[R] b) = domCoprod a b :=
  rfl


/-- When passed an `Equiv.sumCongr`, `MultilinearMap.domDomCongr` distributes over
`MultilinearMap.domCoprod`. -/
theorem domCoprod_domDomCongr_sumCongr (a : MultilinearMap R (fun _ : ι₁ => N) N₁)
    (b : MultilinearMap R (fun _ : ι₂ => N) N₂) (σa : ι₁ ≃ ι₃) (σb : ι₂ ≃ ι₄) :
    (a.domCoprod b).domDomCongr (σa.sumCongr σb) =
      (a.domDomCongr σa).domCoprod (b.domDomCongr σb) :=
  rfl


