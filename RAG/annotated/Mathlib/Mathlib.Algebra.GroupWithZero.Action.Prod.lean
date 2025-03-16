theorem smul_zero_mk {α : Type*} [Monoid M] [AddMonoid α] [DistribMulAction M α] (a : M) (c : β) :
                                        /-
                                          M : Type u_1
                                          β : Type u_4
                                          inst✝³ : SMul M β
                                          α : Type u_5
                                          inst✝² : Monoid M
                                          inst✝¹ : AddMonoid α
                                          inst✝ : DistribMulAction M α
                                          a : M
                                          c : β
                                          ⊢ Eq (HSMul.hSMul a { fst := 0, snd := c }) { fst := 0, snd := HSMul.hSMul a c }
                                        -/
    a • ((0 : α), c) = (0, a • c) := by rw [Prod.smul_mk, smul_zero]
                                        /-
                                          🎉 no goals
                                        -/


theorem smul_mk_zero {β : Type*} [Monoid M] [AddMonoid β] [DistribMulAction M β] (a : M) (b : α) :
                                        /-
                                          M : Type u_1
                                          α : Type u_3
                                          inst✝³ : SMul M α
                                          β : Type u_5
                                          inst✝² : Monoid M
                                          inst✝¹ : AddMonoid β
                                          inst✝ : DistribMulAction M β
                                          a : M
                                          b : α
                                          ⊢ Eq (HSMul.hSMul a { fst := b, snd := 0 }) { fst := HSMul.hSMul a b, snd := 0 }
                                        -/
    a • (b, (0 : β)) = (a • b, 0) := by rw [Prod.smul_mk, smul_zero]
                                        /-
                                          🎉 no goals
                                        -/


instance smulZeroClass {R M N : Type*} [Zero M] [Zero N] [SMulZeroClass R M] [SMulZeroClass R N] :
    SMulZeroClass R (M × N) where smul_zero _ := mk.inj_iff.mpr ⟨smul_zero _, smul_zero _⟩


instance distribSMul {R M N : Type*} [AddZeroClass M] [AddZeroClass N] [DistribSMul R M]
    [DistribSMul R N] : DistribSMul R (M × N) where
  smul_add _ _ _ := mk.inj_iff.mpr ⟨smul_add _ _ _, smul_add _ _ _⟩


instance distribMulAction {R : Type*} [Monoid R] [AddMonoid M] [AddMonoid N]
    [DistribMulAction R M] [DistribMulAction R N] : DistribMulAction R (M × N) :=
  { Prod.mulAction, Prod.distribSMul with }


instance mulDistribMulAction {R : Type*} [Monoid R] [Monoid M] [Monoid N]
    [MulDistribMulAction R M] [MulDistribMulAction R N] : MulDistribMulAction R (M × N) where
  smul_mul _ _ _ := mk.inj_iff.mpr ⟨smul_mul' _ _ _, smul_mul' _ _ _⟩
  smul_one _ := mk.inj_iff.mpr ⟨smul_one _, smul_one _⟩


/-- Construct a `DistribMulAction` by a product monoid from `DistribMulAction`s by the factors. -/
abbrev DistribMulAction.prodOfSMulCommClass [DistribMulAction M α] [DistribMulAction N α]
    [SMulCommClass M N α] : DistribMulAction (M × N) α where
  __ := MulAction.prodOfSMulCommClass M N α
                     /-
                       M : Type u_1
                       N : Type u_2
                       α : Type u_3
                       β : Type u_4
                       inst✝⁵ : Monoid M
                       inst✝⁴ : Monoid N
                       inst✝³ : AddMonoid α
                       inst✝² : DistribMulAction M α
                       inst✝¹ : DistribMulAction N α
                       inst✝ : SMulCommClass M N α
                       mn : Prod M N
                       ⊢ Eq (HSMul.hSMul mn 0) 0
                     -/
  smul_zero mn := by change mn.1 • mn.2 • 0 = (0 : α); rw [smul_zero, smul_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/
                         /-
                           M : Type u_1
                           N : Type u_2
                           α : Type u_3
                           β : Type u_4
                           inst✝⁵ : Monoid M
                           inst✝⁴ : Monoid N
                           inst✝³ : AddMonoid α
                           inst✝² : DistribMulAction M α
                           inst✝¹ : DistribMulAction N α
                           inst✝ : SMulCommClass M N α
                           mn : Prod M N
                           a a' : α
                           ⊢ Eq (HSMul.hSMul mn (HAdd.hAdd a a')) (HAdd.hAdd (HSMul.hSMul mn a) (HSMul.hS …
                         -/
  smul_add mn a a' := by change mn.1 • mn.2 • _ = (_ : α); rw [smul_add, smul_add]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


/-- A `DistribMulAction` by a product monoid is equivalent to
  commuting `DistribMulAction`s by the factors. -/
def DistribMulAction.prodEquiv : DistribMulAction (M × N) α ≃
    Σ' (_ : DistribMulAction M α) (_ : DistribMulAction N α), SMulCommClass M N α where
  toFun _ :=
    letI instM := DistribMulAction.compHom α (.inl M N)
    letI instN := DistribMulAction.compHom α (.inr M N)
    ⟨instM, instN, (MulAction.prodEquiv M N α inferInstance).2.2⟩
  invFun _insts :=
    letI := _insts.1; letI := _insts.2.1; have := _insts.2.2
    DistribMulAction.prodOfSMulCommClass M N α
  left_inv _ := by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      x✝ : DistribMulAction (Prod M N) α
      ⊢ Eq ((fun _insts => letFun ⋯ fun this => DistribMulAction.prodOfSMulCommClass …
    -/
    dsimp only; ext ⟨m, n⟩ a
    /-
      case smul.h.mk.h
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      x✝ : DistribMulAction (Prod M N) α
      m : M
      n : N
      a : α
      ⊢ Eq (SMul.smul { fst := m, snd := n } a) (SMul.smul { fst := m, snd := n } a)
    -/
    change (m, (1 : N)) • ((1 : M), n) • a = _
    /-
      case smul.h.mk.h
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      x✝ : DistribMulAction (Prod M N) α
      m : M
      n : N
      a : α
      ⊢ Eq (HSMul.hSMul { fst := m, snd := 1 } (HSMul.hSMul { fst := 1, snd := n } a …
    -/
    rw [smul_smul, Prod.mk_mul_mk, mul_one, one_mul]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/
  right_inv := by
    /-
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      ⊢ Function.RightInverse (fun _insts => letFun ⋯ fun this => DistribMulAction.p …
    -/
    rintro ⟨_, x, _⟩
    /-
      case mk.mk
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      fst✝ : DistribMulAction M α
      x : DistribMulAction N α
      snd✝ : SMulCommClass M N α
      ⊢ Eq ((fun x => ⟨DistribMulAction.compHom α (MonoidHom.inl M N), ⟨DistribMulAc …
    -/
    dsimp only; congr 1
      /-
        case mk.mk.h.e_3.h
        M : Type u_1
        N : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝² : Monoid M
        inst✝¹ : Monoid N
        inst✝ : AddMonoid α
        fst✝ : DistribMulAction M α
        x : DistribMulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (DistribMulAction.compHom α (MonoidHom.inl M N)) fst✝
      -/
    · ext m a; (conv_rhs => rw [← one_smul N a]); rfl
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case mk.mk.h.e_4
      M : Type u_1
      N : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      inst✝ : AddMonoid α
      fst✝ : DistribMulAction M α
      x : DistribMulAction N α
      snd✝ : SMulCommClass M N α
      ⊢ HEq ⟨DistribMulAction.compHom α (MonoidHom.inr M N), ⋯⟩ ⟨x, snd✝⟩
    -/
    congr 1
      /-
        case mk.mk.h.e_4.e_2.h
        M : Type u_1
        N : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝² : Monoid M
        inst✝¹ : Monoid N
        inst✝ : AddMonoid α
        fst✝ : DistribMulAction M α
        x : DistribMulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (fun x_1 => SMulCommClass M N α) fun x => SMulCommClass M N α
      -/
    · funext i; congr; ext m a; clear i; (conv_rhs => rw [← one_smul N a]); rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
      /-
        case mk.mk.h.e_4.e_3.h
        M : Type u_1
        N : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝² : Monoid M
        inst✝¹ : Monoid N
        inst✝ : AddMonoid α
        fst✝ : DistribMulAction M α
        x : DistribMulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (DistribMulAction.compHom α (MonoidHom.inr M N)) x
      -/
    · ext n a; (conv_rhs => rw [← one_smul M (SMul.smul n a)]); rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/
      /-
        case mk.mk.h.e_4.e_4
        M : Type u_1
        N : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝² : Monoid M
        inst✝¹ : Monoid N
        inst✝ : AddMonoid α
        fst✝ : DistribMulAction M α
        x : DistribMulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ HEq ⋯ snd✝
      -/
    · exact proof_irrel_heq ..
      /-
        🎉 no goals
      -/


