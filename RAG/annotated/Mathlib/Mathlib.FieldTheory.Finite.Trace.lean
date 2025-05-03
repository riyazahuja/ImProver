/-- The trace map from a finite field to its prime field is nongedenerate. -/
theorem trace_to_zmod_nondegenerate (F : Type*) [Field F] [Finite F]
    [Algebra (ZMod (ringChar F)) F] {a : F} (ha : a ≠ 0) :
    ∃ b : F, Algebra.trace (ZMod (ringChar F)) F (a * b) ≠ 0 := by
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Algebra (ZMod (ringChar F)) F
    a : F
    ha : Ne a 0
    ⊢ Exists fun b => Ne ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a b)) 0
  -/
  haveI : Fact (ringChar F).Prime := ⟨CharP.char_is_prime F _⟩
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Algebra (ZMod (ringChar F)) F
    a : F
    ha : Ne a 0
    this : Fact (Nat.Prime (ringChar F))
    ⊢ Exists fun b => Ne ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a b)) 0
  -/
  have htr := traceForm_nondegenerate (ZMod (ringChar F)) F a
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Algebra (ZMod (ringChar F)) F
    a : F
    ha : Ne a 0
    this : Fact (Nat.Prime (ringChar F))
    htr : (∀ (n : F), Eq (((Algebra.traceForm (ZMod (ringChar F)) F) a) n) 0) → Eq …
    ⊢ Exists fun b => Ne ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a b)) 0
  -/
  simp_rw [Algebra.traceForm_apply] at htr
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Algebra (ZMod (ringChar F)) F
    a : F
    ha : Ne a 0
    this : Fact (Nat.Prime (ringChar F))
    htr : (∀ (n : F), Eq ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a n)) 0 …
    ⊢ Exists fun b => Ne ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a b)) 0
  -/
  by_contra! hf
  /-
    F : Type u_1
    inst✝² : Field F
    inst✝¹ : Finite F
    inst✝ : Algebra (ZMod (ringChar F)) F
    a : F
    ha : Ne a 0
    this : Fact (Nat.Prime (ringChar F))
    htr : (∀ (n : F), Eq ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a n)) 0 …
    hf : ∀ (b : F), Eq ((Algebra.trace (ZMod (ringChar F)) F) (HMul.hMul a b)) 0
    ⊢ False
  -/
  exact ha (htr hf)
  /-
    🎉 no goals
  -/


