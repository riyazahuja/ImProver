@[to_additive] instance smul : SMul M (α × β) where smul a p := (a • p.1, a • p.2)


@[to_additive (attr := simp)] lemma smul_fst : (a • x).1 = a • x.1 := rfl


@[to_additive (attr := simp)] lemma smul_snd : (a • x).2 = a • x.2 := rfl


@[to_additive (attr := simp)]
lemma smul_mk (a : M) (b : α) (c : β) : a • (b, c) = (a • b, a • c) := rfl


@[to_additive]
lemma smul_def (a : M) (x : α × β) : a • x = (a • x.1, a • x.2) := rfl


@[to_additive (attr := simp)] lemma smul_swap : (a • x).swap = a • x.swap := rfl


@[to_additive existing smul]
instance pow : Pow (α × β) E where pow p c := (p.1 ^ c, p.2 ^ c)


@[to_additive existing (attr := simp) (reorder := 6 7) smul_fst]
lemma pow_fst (p : α × β) (c : E) : (p ^ c).fst = p.fst ^ c := rfl


@[to_additive existing (attr := simp) (reorder := 6 7) smul_snd]
lemma pow_snd (p : α × β) (c : E) : (p ^ c).snd = p.snd ^ c := rfl

/- Note that the `c` arguments to this lemmas cannot be in the more natural right-most positions due
to limitations in `to_additive` and `to_additive_reorder`, which will silently fail to reorder more
than two adjacent arguments -/

@[to_additive existing (attr := simp) (reorder := 6 7) smul_mk]
lemma pow_mk (c : E) (a : α) (b : β) : Prod.mk a b ^ c = Prod.mk (a ^ c) (b ^ c) := rfl


@[to_additive existing (reorder := 6 7) smul_def]
lemma pow_def (p : α × β) (c : E) : p ^ c = (p.1 ^ c, p.2 ^ c) := rfl


@[to_additive existing (attr := simp) (reorder := 6 7) smul_swap]
lemma pow_swap (p : α × β) (c : E) : (p ^ c).swap = p.swap ^ c := rfl


@[to_additive vaddAssocClass]
instance isScalarTower [SMul M N] [IsScalarTower M N α] [IsScalarTower M N β] :
    IsScalarTower M N (α × β) where
  smul_assoc _ _ _ := mk.inj_iff.mpr ⟨smul_assoc _ _ _, smul_assoc _ _ _⟩


@[to_additive]
instance smulCommClass [SMulCommClass M N α] [SMulCommClass M N β] : SMulCommClass M N (α × β) where
  smul_comm _ _ _ := mk.inj_iff.mpr ⟨smul_comm _ _ _, smul_comm _ _ _⟩


@[to_additive]
instance isCentralScalar [SMul Mᵐᵒᵖ α] [SMul Mᵐᵒᵖ β] [IsCentralScalar M α] [IsCentralScalar M β] :
    IsCentralScalar M (α × β) where
  op_smul_eq_smul _ _ := Prod.ext (op_smul_eq_smul _ _) (op_smul_eq_smul _ _)


@[to_additive]
instance faithfulSMulLeft [FaithfulSMul M α] [Nonempty β] : FaithfulSMul M (α × β) where
  eq_of_smul_eq_smul h :=
    let ⟨b⟩ := ‹Nonempty β›
                                       /-
                                         M : Type u_1
                                         N : Type u_2
                                         P : Type u_3
                                         E : Type u_4
                                         α : Type u_5
                                         β : Type u_6
                                         inst✝⁷ : SMul M α
                                         inst✝⁶ : SMul M β
                                         inst✝⁵ : SMul N α
                                         inst✝⁴ : SMul N β
                                         a✝ : M
                                         x : Prod α β
                                         inst✝³ : Pow α E
                                         inst✝² : Pow β E
                                         inst✝¹ : FaithfulSMul M α
                                         inst✝ : Nonempty β
                                         m₁✝ m₂✝ : M
                                         h : ∀ (a : Prod α β), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                         b : β
                                         a : α
                                         ⊢ Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                       -/
    eq_of_smul_eq_smul fun a : α => by injection h (a, b)
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive]
instance faithfulSMulRight [Nonempty α] [FaithfulSMul M β] : FaithfulSMul M (α × β) where
  eq_of_smul_eq_smul h :=
    let ⟨a⟩ := ‹Nonempty α›
                                       /-
                                         M : Type u_1
                                         N : Type u_2
                                         P : Type u_3
                                         E : Type u_4
                                         α : Type u_5
                                         β : Type u_6
                                         inst✝⁷ : SMul M α
                                         inst✝⁶ : SMul M β
                                         inst✝⁵ : SMul N α
                                         inst✝⁴ : SMul N β
                                         a✝ : M
                                         x : Prod α β
                                         inst✝³ : Pow α E
                                         inst✝² : Pow β E
                                         inst✝¹ : Nonempty α
                                         inst✝ : FaithfulSMul M β
                                         m₁✝ m₂✝ : M
                                         h : ∀ (a : Prod α β), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ a)
                                         a : α
                                         b : β
                                         ⊢ Eq (HSMul.hSMul m₁✝ b) (HSMul.hSMul m₂✝ b)
                                       -/
    eq_of_smul_eq_smul fun b : β => by injection h (a, b)
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive]
instance smulCommClassBoth [Mul N] [Mul P] [SMul M N] [SMul M P] [SMulCommClass M N N]
    [SMulCommClass M P P] : SMulCommClass M (N × P) (N × P) where
                        /-
                          M : Type u_1
                          N : Type u_2
                          P : Type u_3
                          E : Type u_4
                          α : Type u_5
                          β : Type u_6
                          inst✝⁵ : Mul N
                          inst✝⁴ : Mul P
                          inst✝³ : SMul M N
                          inst✝² : SMul M P
                          inst✝¹ : SMulCommClass M N N
                          inst✝ : SMulCommClass M P P
                          c : M
                          x y : Prod N P
                          ⊢ Eq (HSMul.hSMul c (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSMul c y))
                        -/
  smul_comm c x y := by simp [smul_def, mul_def, mul_smul_comm]
                        /-
                          🎉 no goals
                        -/


instance isScalarTowerBoth [Mul N] [Mul P] [SMul M N] [SMul M P] [IsScalarTower M N N]
    [IsScalarTower M P P] : IsScalarTower M (N × P) (N × P) where
                         /-
                           M : Type u_1
                           N : Type u_2
                           P : Type u_3
                           E : Type u_4
                           α : Type u_5
                           β : Type u_6
                           inst✝⁵ : Mul N
                           inst✝⁴ : Mul P
                           inst✝³ : SMul M N
                           inst✝² : SMul M P
                           inst✝¹ : IsScalarTower M N N
                           inst✝ : IsScalarTower M P P
                           c : M
                           x y : Prod N P
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul c x) y) (HSMul.hSMul c (HSMul.hSMul x y))
                         -/
  smul_assoc c x y := by simp [smul_def, mul_def, smul_mul_assoc]
                         /-
                           🎉 no goals
                         -/


@[to_additive]
instance mulAction [Monoid M] [MulAction M α] [MulAction M β] : MulAction M (α × β) where
  mul_smul _ _ _ := mk.inj_iff.mpr ⟨mul_smul _ _ _, mul_smul _ _ _⟩
  one_smul _ := mk.inj_iff.mpr ⟨one_smul _ _, one_smul _ _⟩


/-- Scalar multiplication as a multiplicative homomorphism. -/
@[simps]
def smulMulHom [Monoid α] [Mul β] [MulAction α β] [IsScalarTower α β β] [SMulCommClass α β β] :
    α × β →ₙ* β where
  toFun a := a.1 • a.2
  map_mul' _ _ := (smul_mul_smul_comm _ _ _ _).symm


/-- Scalar multiplication as a monoid homomorphism. -/
@[simps]
def smulMonoidHom [Monoid α] [MulOneClass β] [MulAction α β] [IsScalarTower α β β]
    [SMulCommClass α β β] : α × β →* β :=
  { smulMulHom with map_one' := one_smul _ _ }


/-- Construct a `MulAction` by a product monoid from `MulAction`s by the factors.
  This is not an instance to avoid diamonds for example when `α := M × N`. -/
@[to_additive AddAction.prodOfVAddCommClass
"Construct an `AddAction` by a product monoid from `AddAction`s by the factors.
This is not an instance to avoid diamonds for example when `α := M × N`."]
abbrev MulAction.prodOfSMulCommClass [MulAction M α] [MulAction N α] [SMulCommClass M N α] :
    MulAction (M × N) α where
  smul mn a := mn.1 • mn.2 • a
  one_smul a := (one_smul M _).trans (one_smul N a)
  mul_smul x y a := by
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁴ : Monoid M
      inst✝³ : Monoid N
      inst✝² : MulAction M α
      inst✝¹ : MulAction N α
      inst✝ : SMulCommClass M N α
      x y : Prod M N
      a : α
      ⊢ Eq (HSMul.hSMul (HMul.hMul x y) a) (HSMul.hSMul x (HSMul.hSMul y a))
    -/
    change (x.1 * y.1) • (x.2 * y.2) • a = x.1 • x.2 • y.1 • y.2 • a
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁴ : Monoid M
      inst✝³ : Monoid N
      inst✝² : MulAction M α
      inst✝¹ : MulAction N α
      inst✝ : SMulCommClass M N α
      x y : Prod M N
      a : α
      ⊢ Eq (HSMul.hSMul (HMul.hMul x.1 y.1) (HSMul.hSMul (HMul.hMul x.2 y.2) a)) (HS …
    -/
    rw [mul_smul, mul_smul, smul_comm y.1 x.2]
    /-
      🎉 no goals
    -/


/-- A `MulAction` by a product monoid is equivalent to commuting `MulAction`s by the factors. -/
@[to_additive AddAction.prodEquiv
"An `AddAction` by a product monoid is equivalent to commuting `AddAction`s by the factors."]
def MulAction.prodEquiv :
    MulAction (M × N) α ≃ Σ' (_ : MulAction M α) (_ : MulAction N α), SMulCommClass M N α where
  toFun _ :=
    letI instM := MulAction.compHom α (.inl M N)
    letI instN := MulAction.compHom α (.inr M N)
    ⟨instM, instN,
    { smul_comm := fun m n a ↦ by
        /-
          M : Type u_1
          N : Type u_2
          P : Type u_3
          E : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : Monoid M
          inst✝ : Monoid N
          x✝ : MulAction (Prod M N) α
          instM : MulAction M α := MulAction.compHom α (MonoidHom.inl M N)
          instN : MulAction N α := MulAction.compHom α (MonoidHom.inr M N)
          m : M
          n : N
          a : α
          ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n a)) (HSMul.hSMul n (HSMul.hSMul m a))
        -/
        change (m, (1 : N)) • ((1 : M), n) • a = ((1 : M), n) • (m, (1 : N)) • a
        /-
          M : Type u_1
          N : Type u_2
          P : Type u_3
          E : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝¹ : Monoid M
          inst✝ : Monoid N
          x✝ : MulAction (Prod M N) α
          instM : MulAction M α := MulAction.compHom α (MonoidHom.inl M N)
          instN : MulAction N α := MulAction.compHom α (MonoidHom.inr M N)
          m : M
          n : N
          a : α
          ⊢ Eq (HSMul.hSMul { fst := m, snd := 1 } (HSMul.hSMul { fst := 1, snd := n } a …
        -/
        simp_rw [smul_smul, Prod.mk_mul_mk, mul_one, one_mul] }⟩
        /-
          🎉 no goals
        -/
  invFun _insts :=
    letI := _insts.1; letI := _insts.2.1; have := _insts.2.2
    MulAction.prodOfSMulCommClass M N α
  left_inv := by
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      ⊢ Function.LeftInverse (fun _insts => letFun ⋯ fun this => MulAction.prodOfSMu …
    -/
    rintro ⟨-, hsmul⟩; dsimp only; ext ⟨m, n⟩ a
    /-
      case mk.smul.h.mk.h
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      toSMul✝ : SMul (Prod M N) α
      one_smul✝ : ∀ (b : α), Eq (HSMul.hSMul 1 b) b
      hsmul : ∀ (x y : Prod M N) (b : α), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul. …
      m : M
      n : N
      a : α
      ⊢ Eq (SMul.smul { fst := m, snd := n } a) (SMul.smul { fst := m, snd := n } a)
    -/
    change (m, (1 : N)) • ((1 : M), n) • a = _
    /-
      case mk.smul.h.mk.h
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      toSMul✝ : SMul (Prod M N) α
      one_smul✝ : ∀ (b : α), Eq (HSMul.hSMul 1 b) b
      hsmul : ∀ (x y : Prod M N) (b : α), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul. …
      m : M
      n : N
      a : α
      ⊢ Eq (HSMul.hSMul { fst := m, snd := 1 } (HSMul.hSMul { fst := 1, snd := n } a …
    -/
    rw [← hsmul, Prod.mk_mul_mk, mul_one, one_mul]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  right_inv := by
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      ⊢ Function.RightInverse (fun _insts => letFun ⋯ fun this => MulAction.prodOfSM …
    -/
    rintro ⟨hM, hN, -⟩
    /-
      case mk.mk
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      hM : MulAction M α
      hN : MulAction N α
      snd✝ : SMulCommClass M N α
      ⊢ Eq ((fun x => ⟨MulAction.compHom α (MonoidHom.inl M N), ⟨MulAction.compHom α …
    -/
    dsimp only; congr 1
      /-
        case mk.mk.h.e_3.h
        M : Type u_1
        N : Type u_2
        P : Type u_3
        E : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : Monoid M
        inst✝ : Monoid N
        hM : MulAction M α
        hN : MulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (MulAction.compHom α (MonoidHom.inl M N)) hM
      -/
    · ext m a; (conv_rhs => rw [← hN.one_smul a]); rfl
                                                   /-
                                                     🎉 no goals
                                                   -/
    /-
      case mk.mk.h.e_4
      M : Type u_1
      N : Type u_2
      P : Type u_3
      E : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝¹ : Monoid M
      inst✝ : Monoid N
      hM : MulAction M α
      hN : MulAction N α
      snd✝ : SMulCommClass M N α
      ⊢ HEq ⟨MulAction.compHom α (MonoidHom.inr M N), ⋯⟩ ⟨hN, snd✝⟩
    -/
    congr 1
      /-
        case mk.mk.h.e_4.e_2.h
        M : Type u_1
        N : Type u_2
        P : Type u_3
        E : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : Monoid M
        inst✝ : Monoid N
        hM : MulAction M α
        hN : MulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (fun x => SMulCommClass M N α) fun x => SMulCommClass M N α
      -/
    · funext; congr; ext m a; (conv_rhs => rw [← hN.one_smul a]); rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
      /-
        case mk.mk.h.e_4.e_3.h
        M : Type u_1
        N : Type u_2
        P : Type u_3
        E : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : Monoid M
        inst✝ : Monoid N
        hM : MulAction M α
        hN : MulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ Eq (MulAction.compHom α (MonoidHom.inr M N)) hN
      -/
    · ext n a; (conv_rhs => rw [← hM.one_smul (SMul.smul n a)]); rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      /-
        case mk.mk.h.e_4.e_4
        M : Type u_1
        N : Type u_2
        P : Type u_3
        E : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝¹ : Monoid M
        inst✝ : Monoid N
        hM : MulAction M α
        hN : MulAction N α
        snd✝ : SMulCommClass M N α
        ⊢ HEq ⋯ snd✝
      -/
    · exact proof_irrel_heq ..
      /-
        🎉 no goals
      -/


