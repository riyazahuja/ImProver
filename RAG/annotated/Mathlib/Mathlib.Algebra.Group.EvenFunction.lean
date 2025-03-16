/-- A function `f` is _even_ if it satisfies `f (-x) = f x` for all `x`. -/
protected def Even (f : α → β) : Prop := ∀ a, f (-a) = f a


/-- A function `f` is _odd_ if it satisfies `f (-x) = -f x` for all `x`. -/
protected def Odd [Neg β] (f : α → β) : Prop := ∀ a, f (-a) = -(f a)


/-- Any constant function is even. -/
lemma Even.const (b : β) : Function.Even (fun _ : α ↦ b) := fun _ ↦ rfl


/-- The zero function is even. -/
lemma Even.zero [Zero β] : Function.Even (fun (_ : α) ↦ (0 : β)) := Even.const 0


/-- The zero function is odd. -/
lemma Odd.zero [NegZeroClass β] : Function.Odd (fun (_ : α) ↦ (0 : β)) := fun _ ↦ neg_zero.symm


/-- If `f` is arbitrary and `g` is even, then `f ∘ g` is even. -/
lemma Even.left_comp {g : α → β} (hg : g.Even) (f : β → γ) : (f ∘ g).Even :=
  (congr_arg f <| hg ·)


/-- If `f` is even and `g` is odd, then `f ∘ g` is even. -/
lemma Even.comp_odd [Neg β] {f : β → γ} (hf : f.Even) {g : α → β} (hg : g.Odd) :
    (f ∘ g).Even := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    inst✝ : Neg β
    f : β → γ
    hf : Function.Even f
    g : α → β
    hg : Function.Odd g
    ⊢ Function.Even (Function.comp f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    inst✝ : Neg β
    f : β → γ
    hf : Function.Even f
    g : α → β
    hg : Function.Odd g
    a : α
    ⊢ Eq (Function.comp f g (Neg.neg a)) (Function.comp f g a)
  -/
  simp only [comp_apply, hg a, hf _]
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are odd, then `f ∘ g` is odd. -/
lemma Odd.comp_odd [Neg β] [Neg γ] {f : β → γ} (hf : f.Odd) {g : α → β} (hg : g.Odd) :
    (f ∘ g).Odd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Neg α
    γ : Type u_3
    inst✝¹ : Neg β
    inst✝ : Neg γ
    f : β → γ
    hf : Function.Odd f
    g : α → β
    hg : Function.Odd g
    ⊢ Function.Odd (Function.comp f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Neg α
    γ : Type u_3
    inst✝¹ : Neg β
    inst✝ : Neg γ
    f : β → γ
    hf : Function.Odd f
    g : α → β
    hg : Function.Odd g
    a : α
    ⊢ Eq (Function.comp f g (Neg.neg a)) (Neg.neg (Function.comp f g a))
  -/
  simp only [comp_apply, hg a, hf _]
  /-
    🎉 no goals
  -/


lemma Even.add [Add β] {f g : α → β} (hf : f.Even) (hg : g.Even) : (f + g).Even := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    inst✝ : Add β
    f g : α → β
    hf : Function.Even f
    hg : Function.Even g
    ⊢ Function.Even (HAdd.hAdd f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    inst✝ : Add β
    f g : α → β
    hf : Function.Even f
    hg : Function.Even g
    a : α
    ⊢ Eq (HAdd.hAdd f g (Neg.neg a)) (HAdd.hAdd f g a)
  -/
  simp only [hf a, hg a, Pi.add_apply]
  /-
    🎉 no goals
  -/


lemma Odd.add [SubtractionCommMonoid β] {f g : α → β} (hf : f.Odd) (hg : g.Odd) : (f + g).Odd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    inst✝ : SubtractionCommMonoid β
    f g : α → β
    hf : Function.Odd f
    hg : Function.Odd g
    ⊢ Function.Odd (HAdd.hAdd f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    inst✝ : SubtractionCommMonoid β
    f g : α → β
    hf : Function.Odd f
    hg : Function.Odd g
    a : α
    ⊢ Eq (HAdd.hAdd f g (Neg.neg a)) (Neg.neg (HAdd.hAdd f g a))
  -/
  simp only [hf a, hg a, Pi.add_apply, neg_add]
  /-
    🎉 no goals
  -/


lemma Even.smul_even [SMul β γ] (hf : f.Even) (hg : g.Even) : (f • g).Even := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝ : SMul β γ
    hf : Function.Even f
    hg : Function.Even g
    ⊢ Function.Even (HSMul.hSMul f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝ : SMul β γ
    hf : Function.Even f
    hg : Function.Even g
    a : α
    ⊢ Eq (HSMul.hSMul f g (Neg.neg a)) (HSMul.hSMul f g a)
  -/
  simp only [Pi.smul_apply', hf a, hg a]
  /-
    🎉 no goals
  -/


lemma Even.smul_odd [Monoid β] [AddGroup γ] [DistribMulAction β γ] (hf : f.Even) (hg : g.Odd) :
    (f • g).Odd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Monoid β
    inst✝¹ : AddGroup γ
    inst✝ : DistribMulAction β γ
    hf : Function.Even f
    hg : Function.Odd g
    ⊢ Function.Odd (HSMul.hSMul f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Monoid β
    inst✝¹ : AddGroup γ
    inst✝ : DistribMulAction β γ
    hf : Function.Even f
    hg : Function.Odd g
    a : α
    ⊢ Eq (HSMul.hSMul f g (Neg.neg a)) (Neg.neg (HSMul.hSMul f g a))
  -/
  simp only [Pi.smul_apply', hf a, hg a, smul_neg]
  /-
    🎉 no goals
  -/


lemma Odd.smul_even [Ring β] [AddCommGroup γ] [Module β γ] (hf : f.Odd) (hg : g.Even) :
    (f • g).Odd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Ring β
    inst✝¹ : AddCommGroup γ
    inst✝ : Module β γ
    hf : Function.Odd f
    hg : Function.Even g
    ⊢ Function.Odd (HSMul.hSMul f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Ring β
    inst✝¹ : AddCommGroup γ
    inst✝ : Module β γ
    hf : Function.Odd f
    hg : Function.Even g
    a : α
    ⊢ Eq (HSMul.hSMul f g (Neg.neg a)) (Neg.neg (HSMul.hSMul f g a))
  -/
  simp only [Pi.smul_apply', hf a, hg a, neg_smul]
  /-
    🎉 no goals
  -/


lemma Odd.smul_odd [Ring β] [AddCommGroup γ] [Module β γ] (hf : f.Odd) (hg : g.Odd) :
    (f • g).Even := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Ring β
    inst✝¹ : AddCommGroup γ
    inst✝ : Module β γ
    hf : Function.Odd f
    hg : Function.Odd g
    ⊢ Function.Even (HSMul.hSMul f g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    f : α → β
    g : α → γ
    inst✝² : Ring β
    inst✝¹ : AddCommGroup γ
    inst✝ : Module β γ
    hf : Function.Odd f
    hg : Function.Odd g
    a : α
    ⊢ Eq (HSMul.hSMul f g (Neg.neg a)) (HSMul.hSMul f g a)
  -/
  simp only [Pi.smul_apply', hf a, hg a, smul_neg, neg_smul, neg_neg]
  /-
    🎉 no goals
  -/


lemma Even.const_smul [SMul β γ] (hg : g.Even) (r : β) : (r • g).Even := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    g : α → γ
    inst✝ : SMul β γ
    hg : Function.Even g
    r : β
    ⊢ Function.Even (HSMul.hSMul r g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Neg α
    γ : Type u_3
    g : α → γ
    inst✝ : SMul β γ
    hg : Function.Even g
    r : β
    a : α
    ⊢ Eq (HSMul.hSMul r g (Neg.neg a)) (HSMul.hSMul r g a)
  -/
  simp only [Pi.smul_apply, hg a]
  /-
    🎉 no goals
  -/


lemma Odd.const_smul [Monoid β] [AddGroup γ] [DistribMulAction β γ] (hg : g.Odd) (r : β) :
    (r • g).Odd := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    g : α → γ
    inst✝² : Monoid β
    inst✝¹ : AddGroup γ
    inst✝ : DistribMulAction β γ
    hg : Function.Odd g
    r : β
    ⊢ Function.Odd (HSMul.hSMul r g)
  -/
  intro a
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Neg α
    γ : Type u_3
    g : α → γ
    inst✝² : Monoid β
    inst✝¹ : AddGroup γ
    inst✝ : DistribMulAction β γ
    hg : Function.Odd g
    r : β
    a : α
    ⊢ Eq (HSMul.hSMul r g (Neg.neg a)) (Neg.neg (HSMul.hSMul r g a))
  -/
  simp only [Pi.smul_apply, hg a, smul_neg]
  /-
    🎉 no goals
  -/


lemma Even.mul_even (hf : f.Even) (hg : g.Even) : (f * g).Even := by
  /-
    α : Type u_1
    inst✝¹ : Neg α
    R : Type u_3
    inst✝ : Mul R
    f g : α → R
    hf : Function.Even f
    hg : Function.Even g
    ⊢ Function.Even (HMul.hMul f g)
  -/
  intro a
  /-
    α : Type u_1
    inst✝¹ : Neg α
    R : Type u_3
    inst✝ : Mul R
    f g : α → R
    hf : Function.Even f
    hg : Function.Even g
    a : α
    ⊢ Eq (HMul.hMul f g (Neg.neg a)) (HMul.hMul f g a)
  -/
  simp only [Pi.mul_apply, hf a, hg a]
  /-
    🎉 no goals
  -/


lemma Even.mul_odd [HasDistribNeg R] (hf : f.Even) (hg : g.Odd) : (f * g).Odd := by
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Even f
    hg : Function.Odd g
    ⊢ Function.Odd (HMul.hMul f g)
  -/
  intro a
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Even f
    hg : Function.Odd g
    a : α
    ⊢ Eq (HMul.hMul f g (Neg.neg a)) (Neg.neg (HMul.hMul f g a))
  -/
  simp only [Pi.mul_apply, hf a, hg a, mul_neg]
  /-
    🎉 no goals
  -/


lemma Odd.mul_even [HasDistribNeg R] (hf : f.Odd) (hg : g.Even) : (f * g).Odd := by
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Odd f
    hg : Function.Even g
    ⊢ Function.Odd (HMul.hMul f g)
  -/
  intro a
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Odd f
    hg : Function.Even g
    a : α
    ⊢ Eq (HMul.hMul f g (Neg.neg a)) (Neg.neg (HMul.hMul f g a))
  -/
  simp only [Pi.mul_apply, hf a, hg a, neg_mul]
  /-
    🎉 no goals
  -/


lemma Odd.mul_odd [HasDistribNeg R] (hf : f.Odd) (hg : g.Odd) : (f * g).Even := by
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Odd f
    hg : Function.Odd g
    ⊢ Function.Even (HMul.hMul f g)
  -/
  intro a
  /-
    α : Type u_1
    inst✝² : Neg α
    R : Type u_3
    inst✝¹ : Mul R
    f g : α → R
    inst✝ : HasDistribNeg R
    hf : Function.Odd f
    hg : Function.Odd g
    a : α
    ⊢ Eq (HMul.hMul f g (Neg.neg a)) (HMul.hMul f g a)
  -/
  simp only [Pi.mul_apply, hf a, hg a, mul_neg, neg_mul, neg_neg]
  /-
    🎉 no goals
  -/


/--
If `f` is both even and odd, and its target is a torsion-free commutative additive group,
then `f = 0`.
-/
lemma zero_of_even_and_odd [Neg α] (he : f.Even) (ho : f.Odd) : f = 0 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝² : AddCommGroup β
    inst✝¹ : NoZeroSMulDivisors Nat β
    f : α → β
    inst✝ : Neg α
    he : Function.Even f
    ho : Function.Odd f
    ⊢ Eq f 0
  -/
  ext r
  /-
    case h
    α : Type u_3
    β : Type u_4
    inst✝² : AddCommGroup β
    inst✝¹ : NoZeroSMulDivisors Nat β
    f : α → β
    inst✝ : Neg α
    he : Function.Even f
    ho : Function.Odd f
    r : α
    ⊢ Eq (f r) (0 r)
  -/
  rw [Pi.zero_apply, ← neg_eq_self ℕ, ← ho, he]
  /-
    🎉 no goals
  -/


/-- The sum of the values of an odd function is 0. -/
lemma Odd.sum_eq_zero [Fintype α] [InvolutiveNeg α] {f : α → β} (hf : f.Odd) : ∑ a, f a = 0 := by
  simpa only [neg_eq_self ℕ, Finset.sum_neg_distrib, funext hf, Equiv.neg_apply] using
    Equiv.sum_comp (.neg α) f


/-- An odd function vanishes at zero. -/
lemma Odd.map_zero [NegZeroClass α] (hf : f.Odd) : f 0 = 0 := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝² : AddCommGroup β
    inst✝¹ : NoZeroSMulDivisors Nat β
    f : α → β
    inst✝ : NegZeroClass α
    hf : Function.Odd f
    ⊢ Eq (f 0) 0
  -/
  simp only [← neg_eq_self ℕ, ← hf 0, neg_zero]
  /-
    🎉 no goals
  -/


