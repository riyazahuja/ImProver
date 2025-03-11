instance smulZeroClass [Zero M] [SMulZeroClass R M] : SMulZeroClass R (α →₀ M) where
  smul a v := v.mapRange (a • ·) (smul_zero _)
  smul_zero a := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : Zero M
      inst✝ : SMulZeroClass R M
      a : R
      ⊢ Eq (HSMul.hSMul a 0) 0
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      ι : Type u_4
      M : Type u_5
      M' : Type u_6
      N : Type u_7
      P : Type u_8
      G : Type u_9
      H : Type u_10
      R : Type u_11
      S : Type u_12
      inst✝¹ : Zero M
      inst✝ : SMulZeroClass R M
      a : R
      a✝ : α
      ⊢ Eq ((HSMul.hSMul a 0) a✝) (0 a✝)
    -/
    apply smul_zero
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_smul [Zero M] [SMulZeroClass R M] (b : R) (v : α →₀ M) : ⇑(b • v) = b • ⇑v :=
  rfl


theorem smul_apply [Zero M] [SMulZeroClass R M] (b : R) (v : α →₀ M) (a : α) :
    (b • v) a = b • v a :=
  rfl


instance instSMulWithZero [Zero R] [Zero M] [SMulWithZero R M] : SMulWithZero R (α →₀ M) where
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      ι : Type u_4
                      M : Type u_5
                      M' : Type u_6
                      N : Type u_7
                      P : Type u_8
                      G : Type u_9
                      H : Type u_10
                      R : Type u_11
                      S : Type u_12
                      inst✝² : Zero R
                      inst✝¹ : Zero M
                      inst✝ : SMulWithZero R M
                      f : Finsupp α M
                      ⊢ Eq (HSMul.hSMul 0 f) 0
                    -/
  zero_smul f := by ext i; exact zero_smul _ _
                           /-
                             🎉 no goals
                           -/


instance distribSMul [AddZeroClass M] [DistribSMul R M] : DistribSMul R (α →₀ M) where
  smul := (· • ·)
  smul_add _ _ _ := ext fun _ => smul_add _ _ _
  smul_zero _ := ext fun _ => smul_zero _


instance isScalarTower [Zero M] [SMulZeroClass R M] [SMulZeroClass S M] [SMul R S]
  [IsScalarTower R S M] : IsScalarTower R S (α →₀ M) where
  smul_assoc _ _ _ := ext fun _ => smul_assoc _ _ _


instance smulCommClass [Zero M] [SMulZeroClass R M] [SMulZeroClass S M] [SMulCommClass R S M] :
  SMulCommClass R S (α →₀ M) where
  smul_comm _ _ _ := ext fun _ => smul_comm _ _ _


instance isCentralScalar [Zero M] [SMulZeroClass R M] [SMulZeroClass Rᵐᵒᵖ M] [IsCentralScalar R M] :
  IsCentralScalar R (α →₀ M) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


theorem support_smul [AddMonoid M] [SMulZeroClass R M] {b : R} {g : α →₀ M} :
    (b • g).support ⊆ g.support := fun a => by
  /-
    α : Type u_1
    M : Type u_5
    R : Type u_11
    inst✝¹ : AddMonoid M
    inst✝ : SMulZeroClass R M
    b : R
    g : Finsupp α M
    a : α
    ⊢ Membership.mem (HSMul.hSMul b g).support a → Membership.mem g.support a
  -/
  simp only [smul_apply, mem_support_iff, Ne]
  /-
    α : Type u_1
    M : Type u_5
    R : Type u_11
    inst✝¹ : AddMonoid M
    inst✝ : SMulZeroClass R M
    b : R
    g : Finsupp α M
    a : α
    ⊢ Not (Eq (HSMul.hSMul b (g a)) 0) → Not (Eq (g a) 0)
  -/
  exact mt fun h => h.symm ▸ smul_zero _
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_single [Zero M] [SMulZeroClass R M] (c : R) (a : α) (b : M) :
    c • Finsupp.single a b = Finsupp.single a (c • b) :=
  mapRange_single


theorem mapRange_smul {_ : Monoid R} [AddMonoid M] [DistribMulAction R M] [AddMonoid N]
    [DistribMulAction R N] {f : M → N} {hf : f 0 = 0} (c : R) (v : α →₀ M)
    (hsmul : ∀ x, f (c • x) = c • f x) : mapRange f hf (c • v) = c • mapRange f hf v := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    R : Type u_11
    x✝ : Monoid R
    inst✝³ : AddMonoid M
    inst✝² : DistribMulAction R M
    inst✝¹ : AddMonoid N
    inst✝ : DistribMulAction R N
    f : M → N
    hf : Eq (f 0) 0
    c : R
    v : Finsupp α M
    hsmul : ∀ (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
    ⊢ Eq (Finsupp.mapRange f hf (HSMul.hSMul c v)) (HSMul.hSMul c (Finsupp.mapRang …
  -/
  erw [← mapRange_comp]
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_7
      R : Type u_11
      x✝ : Monoid R
      inst✝³ : AddMonoid M
      inst✝² : DistribMulAction R M
      inst✝¹ : AddMonoid N
      inst✝ : DistribMulAction R N
      f : M → N
      hf : Eq (f 0) 0
      c : R
      v : Finsupp α M
      hsmul : ∀ (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
      ⊢ Eq (Finsupp.mapRange (Function.comp f fun x => HSMul.hSMul c x) ?h v) (HSMul …
    -/
  · have : f ∘ (c • ·) = (c • ·) ∘ f := funext hsmul
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_7
      R : Type u_11
      x✝ : Monoid R
      inst✝³ : AddMonoid M
      inst✝² : DistribMulAction R M
      inst✝¹ : AddMonoid N
      inst✝ : DistribMulAction R N
      f : M → N
      hf : Eq (f 0) 0
      c : R
      v : Finsupp α M
      hsmul : ∀ (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
      this : Eq (Function.comp f fun x => HSMul.hSMul c x) (Function.comp (fun x =>  …
      ⊢ Eq (Finsupp.mapRange (Function.comp f fun x => HSMul.hSMul c x) ?h v) (HSMul …
    -/
    simp_rw [this]
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_7
      R : Type u_11
      x✝ : Monoid R
      inst✝³ : AddMonoid M
      inst✝² : DistribMulAction R M
      inst✝¹ : AddMonoid N
      inst✝ : DistribMulAction R N
      f : M → N
      hf : Eq (f 0) 0
      c : R
      v : Finsupp α M
      hsmul : ∀ (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
      this : Eq (Function.comp f fun x => HSMul.hSMul c x) (Function.comp (fun x =>  …
      ⊢ Eq (Finsupp.mapRange (Function.comp (fun x => HSMul.hSMul c x) f) ⋯ v) (HSMu …
    -/
    apply mapRange_comp
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_1
    M : Type u_5
    N : Type u_7
    R : Type u_11
    x✝ : Monoid R
    inst✝³ : AddMonoid M
    inst✝² : DistribMulAction R M
    inst✝¹ : AddMonoid N
    inst✝ : DistribMulAction R N
    f : M → N
    hf : Eq (f 0) 0
    c : R
    v : Finsupp α M
    hsmul : ∀ (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
    ⊢ Eq (Function.comp f (fun x => HSMul.hSMul c x) 0) 0
  -/
  simp only [Function.comp_apply, smul_zero, hf]
  /-
    🎉 no goals
  -/


