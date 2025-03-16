/-- A Lie subalgebra of a Lie algebra is submodule that is closed under the Lie bracket.
This is a sufficient condition for the subset itself to form a Lie algebra. -/
structure LieSubalgebra extends Submodule R L where
  /-- A Lie subalgebra is closed under Lie bracket. -/
  lie_mem' : ∀ {x y}, x ∈ carrier → y ∈ carrier → ⁅x, y⁆ ∈ carrier


/-- The zero algebra is a subalgebra of any Lie algebra. -/
instance : Zero (LieSubalgebra R L) :=
  ⟨⟨0, @fun x y hx _hy ↦ by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x y : L
      hx : Membership.mem (Submodule.toAddSubmonoid 0).carrier x
      _hy : Membership.mem (Submodule.toAddSubmonoid 0).carrier y
      ⊢ Membership.mem (Submodule.toAddSubmonoid 0).carrier (Bracket.bracket x y)
    -/
    rw [(Submodule.mem_bot R).1 hx, zero_lie]
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x y : L
      hx : Membership.mem (Submodule.toAddSubmonoid 0).carrier x
      _hy : Membership.mem (Submodule.toAddSubmonoid 0).carrier y
      ⊢ Membership.mem (Submodule.toAddSubmonoid 0).carrier 0
    -/
    exact Submodule.zero_mem 0⟩⟩
    /-
      🎉 no goals
    -/


instance : Inhabited (LieSubalgebra R L) :=
  ⟨0⟩


instance : Coe (LieSubalgebra R L) (Submodule R L) :=
  ⟨LieSubalgebra.toSubmodule⟩


instance : SetLike (LieSubalgebra R L) L where
  coe L' := L'.carrier
  coe_injective' L' L'' h := by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' L'' : LieSubalgebra R L
      h : Eq ((fun L' => L'.carrier) L') ((fun L' => L'.carrier) L'')
      ⊢ Eq L' L''
    -/
    rcases L' with ⟨⟨⟩⟩
    /-
      case mk.mk
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L'' : LieSubalgebra R L
      toAddSubmonoid✝ : AddSubmonoid L
      smul_mem'✝ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝.carrier x → Mem …
      lie_mem'✝ : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝, s …
      h : Eq ((fun L' => L'.carrier) { toAddSubmonoid := toAddSubmonoid✝, smul_mem'  …
      ⊢ Eq { toAddSubmonoid := toAddSubmonoid✝, smul_mem' := smul_mem'✝, lie_mem' := …
    -/
    rcases L'' with ⟨⟨⟩⟩
    /-
      case mk.mk.mk.mk
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      toAddSubmonoid✝¹ : AddSubmonoid L
      smul_mem'✝¹ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝¹.carrier x → M …
      lie_mem'✝¹ : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝¹, …
      toAddSubmonoid✝ : AddSubmonoid L
      smul_mem'✝ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝.carrier x → Mem …
      lie_mem'✝ : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝, s …
      h : Eq ((fun L' => L'.carrier) { toAddSubmonoid := toAddSubmonoid✝¹, smul_mem' …
      ⊢ Eq { toAddSubmonoid := toAddSubmonoid✝¹, smul_mem' := smul_mem'✝¹, lie_mem'  …
    -/
    congr
    /-
      case mk.mk.mk.mk.e_toSubmodule.e_toAddSubmonoid
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      toAddSubmonoid✝¹ : AddSubmonoid L
      smul_mem'✝¹ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝¹.carrier x → M …
      lie_mem'✝¹ : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝¹, …
      toAddSubmonoid✝ : AddSubmonoid L
      smul_mem'✝ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝.carrier x → Mem …
      lie_mem'✝ : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝, s …
      h : Eq ((fun L' => L'.carrier) { toAddSubmonoid := toAddSubmonoid✝¹, smul_mem' …
      ⊢ Eq toAddSubmonoid✝¹ toAddSubmonoid✝
    -/
    exact SetLike.coe_injective' h
    /-
      🎉 no goals
    -/


instance : AddSubgroupClass (LieSubalgebra R L) L where
  add_mem := Submodule.add_mem _
  zero_mem L' := L'.zero_mem'
  neg_mem {L'} x hx := show -x ∈ (L' : Submodule R L) from neg_mem hx


/-- A Lie subalgebra forms a new Lie ring. -/
instance lieRing (L' : LieSubalgebra R L) : LieRing L' where
  bracket x y := ⟨⁅x.val, y.val⁆, L'.lie_mem' x.property y.property⟩
  lie_add := by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      ⊢ ∀ (x y z : Subtype fun x => Membership.mem L' x), Eq (Bracket.bracket x (HAd …
    -/
    intros
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq (Bracket.bracket x✝ (HAdd.hAdd y✝ z✝)) (HAdd.hAdd (Bracket.bracket x✝ y✝) …
    -/
    apply SetCoe.ext
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      ⊢ ∀ (x y z : Subtype fun x => Membership.mem L' x), Eq (Bracket.bracket (HAdd. …
    -/
    /-
      case a
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq ↑(Bracket.bracket x✝ (HAdd.hAdd y✝ z✝)) ↑(HAdd.hAdd (Bracket.bracket x✝ y …
    -/
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq (Bracket.bracket (HAdd.hAdd x✝ y✝) z✝) (HAdd.hAdd (Bracket.bracket x✝ z✝) …
    -/
    apply lie_add
    /-
      case a
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq ↑(Bracket.bracket (HAdd.hAdd x✝ y✝) z✝) ↑(HAdd.hAdd (Bracket.bracket x✝ z …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  add_lie := by
    intros
    apply SetCoe.ext
    apply add_lie
  lie_self := by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      ⊢ ∀ (x : Subtype fun x => Membership.mem L' x), Eq (Bracket.bracket x x) 0
    -/
    intros
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq (Bracket.bracket x✝ x✝) 0
    -/
    apply SetCoe.ext
    /-
      case a
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq ↑(Bracket.bracket x✝ x✝) ↑0
    -/
    apply lie_self
    /-
      🎉 no goals
    -/
  leibniz_lie := by
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      ⊢ ∀ (x y z : Subtype fun x => Membership.mem L' x), Eq (Bracket.bracket x (Bra …
    -/
    intros
    /-
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq (Bracket.bracket x✝ (Bracket.bracket y✝ z✝)) (HAdd.hAdd (Bracket.bracket  …
    -/
    apply SetCoe.ext
    /-
      case a
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      L' : LieSubalgebra R L
      x✝ y✝ z✝ : Subtype fun x => Membership.mem L' x
      ⊢ Eq ↑(Bracket.bracket x✝ (Bracket.bracket y✝ z✝)) ↑(HAdd.hAdd (Bracket.bracke …
    -/
    apply leibniz_lie
    /-
      🎉 no goals
    -/


/-- A Lie subalgebra inherits module structures from `L`. -/
instance [SMul R₁ R] [Module R₁ L] [IsScalarTower R₁ R L] (L' : LieSubalgebra R L) : Module R₁ L' :=
  L'.toSubmodule.module'


instance [SMul R₁ R] [SMul R₁ᵐᵒᵖ R] [Module R₁ L] [Module R₁ᵐᵒᵖ L] [IsScalarTower R₁ R L]
    [IsScalarTower R₁ᵐᵒᵖ R L] [IsCentralScalar R₁ L] (L' : LieSubalgebra R L) :
    IsCentralScalar R₁ L' :=
  L'.toSubmodule.isCentralScalar


instance [SMul R₁ R] [Module R₁ L] [IsScalarTower R₁ R L] (L' : LieSubalgebra R L) :
    IsScalarTower R₁ R L' :=
  L'.toSubmodule.isScalarTower


instance (L' : LieSubalgebra R L) [IsNoetherian R L] : IsNoetherian R L' :=
  isNoetherian_submodule' _


instance (L' : LieSubalgebra R L) [IsArtinian R L] : IsArtinian R L' :=
  isArtinian_submodule' _


/-- A Lie subalgebra forms a new Lie algebra. -/
instance lieAlgebra (L' : LieSubalgebra R L) : LieAlgebra R L' where
  lie_smul := by
    { intros
      apply SetCoe.ext
      apply lie_smul }


@[simp]
protected theorem zero_mem : (0 : L) ∈ L' :=
  zero_mem L'


protected theorem add_mem {x y : L} : x ∈ L' → y ∈ L' → (x + y : L) ∈ L' :=
  add_mem


protected theorem sub_mem {x y : L} : x ∈ L' → y ∈ L' → (x - y : L) ∈ L' :=
  sub_mem


theorem smul_mem (t : R) {x : L} (h : x ∈ L') : t • x ∈ L' :=
  (L' : Submodule R L).smul_mem t h


theorem lie_mem {x y : L} (hx : x ∈ L') (hy : y ∈ L') : (⁅x, y⁆ : L) ∈ L' :=
  L'.lie_mem' hx hy


theorem mem_carrier {x : L} : x ∈ L'.carrier ↔ x ∈ (L' : Set L) :=
  Iff.rfl


@[simp]
theorem mem_mk_iff (S : Set L) (h₁ h₂ h₃ h₄) {x : L} :
    x ∈ (⟨⟨⟨⟨S, h₁⟩, h₂⟩, h₃⟩, h₄⟩ : LieSubalgebra R L) ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem mem_toSubmodule {x : L} : x ∈ (L' : Submodule R L) ↔ x ∈ L' :=
  Iff.rfl


@[deprecated (since := "2024-12-30")] alias mem_coe_submodule := mem_toSubmodule


theorem mem_coe {x : L} : x ∈ (L' : Set L) ↔ x ∈ L' :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_bracket (x y : L') : (↑⁅x, y⁆ : L) = ⁅(↑x : L), ↑y⁆ :=
  rfl


theorem ext_iff (x y : L') : x = y ↔ (x : L) = y :=
  Subtype.ext_iff


theorem coe_zero_iff_zero (x : L') : (x : L) = 0 ↔ x = 0 :=
  (ext_iff L' x 0).symm


@[ext]
theorem ext (L₁' L₂' : LieSubalgebra R L) (h : ∀ x, x ∈ L₁' ↔ x ∈ L₂') : L₁' = L₂' :=
  SetLike.ext h


theorem ext_iff' (L₁' L₂' : LieSubalgebra R L) : L₁' = L₂' ↔ ∀ x, x ∈ L₁' ↔ x ∈ L₂' :=
  SetLike.ext_iff


@[simp]
theorem mk_coe (S : Set L) (h₁ h₂ h₃ h₄) :
    ((⟨⟨⟨⟨S, h₁⟩, h₂⟩, h₃⟩, h₄⟩ : LieSubalgebra R L) : Set L) = S :=
  rfl


theorem toSubmodule_mk (p : Submodule R L) (h) :
    (({ p with lie_mem' := h } : LieSubalgebra R L) : Submodule R L) = p := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    p : Submodule R L
    h : ∀ {x y : L}, Membership.mem p.carrier x → Membership.mem p.carrier y → Mem …
    ⊢ Eq { toSubmodule := p, lie_mem' := h }.toSubmodule p
  -/
  cases p
  /-
    case mk
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    toAddSubmonoid✝ : AddSubmonoid L
    smul_mem'✝ : ∀ (c : R) {x : L}, Membership.mem toAddSubmonoid✝.carrier x → Mem …
    h : ∀ {x y : L}, Membership.mem { toAddSubmonoid := toAddSubmonoid✝, smul_mem' …
    ⊢ Eq { toAddSubmonoid := toAddSubmonoid✝, smul_mem' := smul_mem'✝, lie_mem' := …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias coe_to_submodule_mk := toSubmodule_mk


theorem coe_injective : Function.Injective ((↑) : LieSubalgebra R L → Set L) :=
  SetLike.coe_injective


@[norm_cast]
theorem coe_set_eq (L₁' L₂' : LieSubalgebra R L) : (L₁' : Set L) = L₂' ↔ L₁' = L₂' :=
  SetLike.coe_set_eq


theorem toSubmodule_injective : Function.Injective ((↑) : LieSubalgebra R L → Submodule R L) :=
  fun L₁' L₂' h ↦ by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    L₁' L₂' : LieSubalgebra R L
    h : Eq L₁'.toSubmodule L₂'.toSubmodule
    ⊢ Eq L₁' L₂'
  -/
  rw [SetLike.ext'_iff] at h
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    L₁' L₂' : LieSubalgebra R L
    h : Eq ↑L₁'.toSubmodule ↑L₂'.toSubmodule
    ⊢ Eq L₁' L₂'
  -/
  rw [← coe_set_eq]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    L₁' L₂' : LieSubalgebra R L
    h : Eq ↑L₁'.toSubmodule ↑L₂'.toSubmodule
    ⊢ Eq ↑L₁' ↑L₂'
  -/
  exact h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias to_submodule_injective := toSubmodule_injective


@[simp]
theorem toSubmodule_inj (L₁' L₂' : LieSubalgebra R L) :
    (L₁' : Submodule R L) = (L₂' : Submodule R L) ↔ L₁' = L₂' :=
  toSubmodule_injective.eq_iff


@[deprecated (since := "2024-12-30")] alias coe_to_submodule_inj := toSubmodule_inj


@[deprecated (since := "2024-12-29")] alias toSubmodule_eq_iff := toSubmodule_inj


theorem coe_toSubmodule : ((L' : Submodule R L) : Set L) = L' :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_to_submodule := coe_toSubmodule


instance : Bracket L' M where
  bracket x m := ⁅(x : L), m⁆


@[simp]
theorem coe_bracket_of_module (x : L') (m : M) : ⁅x, m⁆ = ⁅(x : L), m⁆ :=
  rfl


instance : IsLieTower L' L M where
  leibniz_lie x y m := leibniz_lie x.val y m


/-- Given a Lie algebra `L` containing a Lie subalgebra `L' ⊆ L`, together with a Lie ring module
`M` of `L`, we may regard `M` as a Lie ring module of `L'` by restriction. -/
instance lieRingModule : LieRingModule L' M where
  add_lie x y m := add_lie (x : L) y m
  lie_add x y m := lie_add (x : L) y m
  leibniz_lie x y m := leibniz_lie x (y : L) m


/-- Given a Lie algebra `L` containing a Lie subalgebra `L' ⊆ L`, together with a Lie module `M` of
`L`, we may regard `M` as a Lie module of `L'` by restriction. -/
instance lieModule [LieModule R L M] : LieModule R L' M where
  smul_lie t x m := by
    /-
      R : Type u
      L : Type v
      inst✝⁹ : CommRing R
      inst✝⁸ : LieRing L
      inst✝⁷ : LieAlgebra R L
      L' : LieSubalgebra R L
      M : Type w
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : LieRingModule L M
      N : Type w₁
      inst✝⁴ : AddCommGroup N
      inst✝³ : LieRingModule L N
      inst✝² : Module R N
      inst✝¹ : Module R M
      inst✝ : LieModule R L M
      t : R
      x : Subtype fun x => Membership.mem L' x
      m : M
      ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
    -/
    rw [coe_bracket_of_module, Submodule.coe_smul_of_tower, smul_lie, coe_bracket_of_module]
    /-
      🎉 no goals
    -/
                       /-
                         R : Type u
                         L : Type v
                         inst✝⁹ : CommRing R
                         inst✝⁸ : LieRing L
                         inst✝⁷ : LieAlgebra R L
                         L' : LieSubalgebra R L
                         M : Type w
                         inst✝⁶ : AddCommGroup M
                         inst✝⁵ : LieRingModule L M
                         N : Type w₁
                         inst✝⁴ : AddCommGroup N
                         inst✝³ : LieRingModule L N
                         inst✝² : Module R N
                         inst✝¹ : Module R M
                         inst✝ : LieModule R L M
                         t : R
                         x : Subtype fun x => Membership.mem L' x
                         m : M
                         ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
                       -/
  lie_smul t x m := by simp only [coe_bracket_of_module, lie_smul]
                       /-
                         🎉 no goals
                       -/


/-- An `L`-equivariant map of Lie modules `M → N` is `L'`-equivariant for any Lie subalgebra
`L' ⊆ L`. -/
def _root_.LieModuleHom.restrictLie (f : M →ₗ⁅R,L⁆ N) (L' : LieSubalgebra R L) : M →ₗ⁅R,L'⁆ N :=
  { (f : M →ₗ[R] N) with map_lie' := @fun x m ↦ f.map_lie (↑x) m }


@[simp]
theorem _root_.LieModuleHom.coe_restrictLie (f : M →ₗ⁅R,L⁆ N) : ⇑(f.restrictLie L') = f :=
  rfl


/-- The embedding of a Lie subalgebra into the ambient space as a morphism of Lie algebras. -/
def incl : L' →ₗ⁅R⁆ L :=
  { (L' : Submodule R L).subtype with
    map_lie' := rfl }


@[simp]
theorem coe_incl : ⇑L'.incl = ((↑) : L' → L) :=
  rfl


/-- The embedding of a Lie subalgebra into the ambient space as a morphism of Lie modules. -/
def incl' : L' →ₗ⁅R,L'⁆ L :=
  { (L' : Submodule R L).subtype with
    map_lie' := rfl }


@[simp]
theorem coe_incl' : ⇑L'.incl' = ((↑) : L' → L) :=
  rfl


/-- The range of a morphism of Lie algebras is a Lie subalgebra. -/
def range : LieSubalgebra R L₂ :=
  { LinearMap.range (f : L →ₗ[R] L₂) with
      lie_mem' := by
        /-
          R : Type u
          L : Type v
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : LieAlgebra R L
          L₂ : Type w
          inst✝¹ : LieRing L₂
          inst✝ : LieAlgebra R L₂
          f : LieHom R L L₂
          ⊢ ∀ {x y : L₂}, Membership.mem __src✝.carrier x → Membership.mem __src✝.carrie …
        -/
        rintro - - ⟨x, rfl⟩ ⟨y, rfl⟩
        /-
          case intro.intro
          R : Type u
          L : Type v
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : LieAlgebra R L
          L₂ : Type w
          inst✝¹ : LieRing L₂
          inst✝ : LieAlgebra R L₂
          f : LieHom R L L₂
          x y : L
          ⊢ Membership.mem __src✝.carrier (Bracket.bracket (↑f x) (↑f y))
        -/
        exact ⟨⁅x, y⁆, f.map_lie x y⟩ }
        /-
          🎉 no goals
        -/


@[simp]
theorem range_coe : (f.range : Set L₂) = Set.range f :=
  LinearMap.range_coe (f : L →ₗ[R] L₂)


@[simp]
theorem mem_range (x : L₂) : x ∈ f.range ↔ ∃ y : L, f y = x :=
  LinearMap.mem_range


theorem mem_range_self (x : L) : f x ∈ f.range :=
  LinearMap.mem_range_self (f : L →ₗ[R] L₂) x


/-- We can restrict a morphism to a (surjective) map to its range. -/
def rangeRestrict : L →ₗ⁅R⁆ f.range :=
  { (f : L →ₗ[R] L₂).rangeRestrict with
    map_lie' := @fun x y ↦ by
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        x y : L
        ⊢ Eq (__src✝.toFun (Bracket.bracket x y)) (Bracket.bracket (__src✝.toFun x) (_ …
      -/
      apply Subtype.ext
      /-
        case a
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        x y : L
        ⊢ Eq ↑(__src✝.toFun (Bracket.bracket x y)) ↑(Bracket.bracket (__src✝.toFun x)  …
      -/
      exact f.map_lie x y }
      /-
        🎉 no goals
      -/


@[simp]
theorem rangeRestrict_apply (x : L) : f.rangeRestrict x = ⟨f x, f.mem_range_self x⟩ :=
  rfl


theorem surjective_rangeRestrict : Function.Surjective f.rangeRestrict := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    ⊢ Function.Surjective ⇑f.rangeRestrict
  -/
  rintro ⟨y, hy⟩
  /-
    case mk
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    y : L₂
    hy : Membership.mem f.range y
    ⊢ Exists fun a => Eq (f.rangeRestrict a) ⟨y, hy⟩
  -/
  rw [mem_range] at hy; obtain ⟨x, rfl⟩ := hy
  /-
    case mk.intro
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    x : L
    hy : Membership.mem f.range (f x)
    ⊢ Exists fun a => Eq (f.rangeRestrict a) ⟨f x, hy⟩
  -/
  use x
  /-
    case h
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    x : L
    hy : Membership.mem f.range (f x)
    ⊢ Eq (f.rangeRestrict x) ⟨f x, hy⟩
  -/
  simp only [Subtype.mk_eq_mk, rangeRestrict_apply]
  /-
    🎉 no goals
  -/


/-- A Lie algebra is equivalent to its range under an injective Lie algebra morphism. -/
noncomputable def equivRangeOfInjective (h : Function.Injective f) : L ≃ₗ⁅R⁆ f.range :=
  LieEquiv.ofBijective f.rangeRestrict
    ⟨fun x y hxy ↦ by
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        h : Function.Injective ⇑f
        x y : L
        hxy : Eq (f.rangeRestrict x) (f.rangeRestrict y)
        ⊢ Eq x y
      -/
      simp only [Subtype.mk_eq_mk, rangeRestrict_apply] at hxy
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        h : Function.Injective ⇑f
        x y : L
        hxy : Eq (f x) (f y)
        ⊢ Eq x y
      -/
      exact h hxy, f.surjective_rangeRestrict⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem equivRangeOfInjective_apply (h : Function.Injective f) (x : L) :
    f.equivRangeOfInjective h x = ⟨f x, mem_range_self f x⟩ :=
  rfl


theorem Submodule.exists_lieSubalgebra_coe_eq_iff (p : Submodule R L) :
    (∃ K : LieSubalgebra R L, ↑K = p) ↔ ∀ x y : L, x ∈ p → y ∈ p → ⁅x, y⁆ ∈ p := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    p : Submodule R L
    ⊢ Iff (Exists fun K => Eq K.toSubmodule p) (∀ (x y : L), Membership.mem p x →  …
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      ⊢ (Exists fun K => Eq K.toSubmodule p) → ∀ (x y : L), Membership.mem p x → Mem …
    -/
  · rintro ⟨K, rfl⟩ _ _
    /-
      case mp.intro
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K : LieSubalgebra R L
      x✝ y✝ : L
      ⊢ Membership.mem K.toSubmodule x✝ → Membership.mem K.toSubmodule y✝ → Membersh …
    -/
    exact K.lie_mem'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      ⊢ (∀ (x y : L), Membership.mem p x → Membership.mem p y → Membership.mem p (Br …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      h : ∀ (x y : L), Membership.mem p x → Membership.mem p y → Membership.mem p (B …
      ⊢ Exists fun K => Eq K.toSubmodule p
    -/
    use { p with lie_mem' := h _ _ }
    /-
      🎉 no goals
    -/


@[simp]
theorem incl_range : K.incl.range = K := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    ⊢ Eq K.incl.range K
  -/
  rw [← toSubmodule_inj]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    ⊢ Eq K.incl.range.toSubmodule K.toSubmodule
  -/
  exact (K : Submodule R L).range_subtype
  /-
    🎉 no goals
  -/


/-- The image of a Lie subalgebra under a Lie algebra morphism is a Lie subalgebra of the
codomain. -/
def map : LieSubalgebra R L₂ :=
  { (K : Submodule R L).map (f : L →ₗ[R] L₂) with
    lie_mem' := @fun x y hx hy ↦ by
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        hx : Membership.mem __src✝.carrier x
        hy : Membership.mem __src✝.carrier y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x y)
      -/
      erw [Submodule.mem_map] at hx
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        hx : Exists fun y => And (Membership.mem K.toSubmodule y) (Eq (↑f y) x)
        hy : Membership.mem __src✝.carrier y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x y)
      -/
      rcases hx with ⟨x', hx', hx⟩
      /-
        case intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        hy : Membership.mem __src✝.carrier y
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x y)
      -/
      rw [← hx]
      /-
        case intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        hy : Membership.mem __src✝.carrier y
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket (↑f x') y)
      -/
      erw [Submodule.mem_map] at hy
      /-
        case intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        hy : Exists fun y_1 => And (Membership.mem K.toSubmodule y_1) (Eq (↑f y_1) y)
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket (↑f x') y)
      -/
      rcases hy with ⟨y', hy', hy⟩
      /-
        case intro.intro.intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        y' : L
        hy' : Membership.mem K.toSubmodule y'
        hy : Eq (↑f y') y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket (↑f x') y)
      -/
      rw [← hy]
      /-
        case intro.intro.intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        y' : L
        hy' : Membership.mem K.toSubmodule y'
        hy : Eq (↑f y') y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket (↑f x') (↑f y'))
      -/
      erw [Submodule.mem_map]
      /-
        case intro.intro.intro.intro
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L₂
        x' : L
        hx' : Membership.mem K.toSubmodule x'
        hx : Eq (↑f x') x
        y' : L
        hy' : Membership.mem K.toSubmodule y'
        hy : Eq (↑f y') y
        ⊢ Exists fun y => And (Membership.mem K.toSubmodule y) (Eq (↑f y) (Bracket.bra …
      -/
      exact ⟨⁅x', y'⁆, K.lie_mem hx' hy', f.map_lie x' y'⟩ }
      /-
        🎉 no goals
      -/


@[simp]
theorem mem_map (x : L₂) : x ∈ K.map f ↔ ∃ y : L, y ∈ K ∧ f y = x :=
  Submodule.mem_map

-- TODO Rename and state for homs instead of equivs.

theorem mem_map_submodule (e : L ≃ₗ⁅R⁆ L₂) (x : L₂) :
    x ∈ K.map (e : L →ₗ⁅R⁆ L₂) ↔ x ∈ (K : Submodule R L).map (e : L →ₗ[R] L₂) :=
  Iff.rfl


/-- The preimage of a Lie subalgebra under a Lie algebra morphism is a Lie subalgebra of the
domain. -/
def comap : LieSubalgebra R L :=
  { (K₂ : Submodule R L₂).comap (f : L →ₗ[R] L₂) with
    lie_mem' := @fun x y hx hy ↦ by
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L
        hx : Membership.mem __src✝.carrier x
        hy : Membership.mem __src✝.carrier y
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x y)
      -/
      suffices ⁅f x, f y⁆ ∈ K₂ by simp [this]
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        x y : L
        hx : Membership.mem __src✝.carrier x
        hy : Membership.mem __src✝.carrier y
        ⊢ Membership.mem K₂ (Bracket.bracket (f x) (f y))
      -/
      exact K₂.lie_mem hx hy }
      /-
        🎉 no goals
      -/


instance : PartialOrder (LieSubalgebra R L) :=
  { PartialOrder.lift ((↑) : LieSubalgebra R L → Set L) coe_injective with
    le := fun N N' ↦ ∀ ⦃x⦄, x ∈ N → x ∈ N' }


theorem le_def : K ≤ K' ↔ (K : Set L) ⊆ K' :=
  Iff.rfl


@[simp]
theorem toSubmodule_le_toSubmodule : (K : Submodule R L) ≤ K' ↔ K ≤ K' :=
  Iff.rfl


@[deprecated (since := "2024-12-30")]
alias coe_submodule_le_coe_submodule := toSubmodule_le_toSubmodule


instance : Bot (LieSubalgebra R L) :=
  ⟨0⟩


@[simp]
theorem bot_coe : ((⊥ : LieSubalgebra R L) : Set L) = {0} :=
  rfl


@[simp]
theorem bot_toSubmodule : ((⊥ : LieSubalgebra R L) : Submodule R L) = ⊥ :=
  rfl


@[deprecated (since := "2024-12-30")] alias bot_coe_submodule := bot_toSubmodule


@[simp]
theorem mem_bot (x : L) : x ∈ (⊥ : LieSubalgebra R L) ↔ x = 0 :=
  mem_singleton_iff


instance : Top (LieSubalgebra R L) :=
  ⟨{ (⊤ : Submodule R L) with lie_mem' := @fun x y _ _ ↦ mem_univ ⁅x, y⁆ }⟩


@[simp]
theorem top_coe : ((⊤ : LieSubalgebra R L) : Set L) = univ :=
  rfl


@[simp]
theorem top_toSubmodule : ((⊤ : LieSubalgebra R L) : Submodule R L) = ⊤ :=
  rfl


@[deprecated (since := "2024-12-30")] alias top_coe_submodule := top_toSubmodule


@[simp]
theorem mem_top (x : L) : x ∈ (⊤ : LieSubalgebra R L) :=
  mem_univ x


theorem _root_.LieHom.range_eq_map : f.range = map f ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    ⊢ Eq f.range (LieSubalgebra.map f Top.top)
  -/
  ext
  /-
    case h
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    x✝ : L₂
    ⊢ Iff (Membership.mem f.range x✝) (Membership.mem (LieSubalgebra.map f Top.top …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Min (LieSubalgebra R L) :=
  ⟨fun K K' ↦
    { (K ⊓ K' : Submodule R L) with
      lie_mem' := fun hx hy ↦ mem_inter (K.lie_mem hx.1 hy.1) (K'.lie_mem hx.2 hy.2) }⟩


instance : InfSet (LieSubalgebra R L) :=
  ⟨fun S ↦
    { sInf {(s : Submodule R L) | s ∈ S} with
      lie_mem' := @fun x y hx hy ↦ by
        simp only [Submodule.mem_carrier, mem_iInter, Submodule.sInf_coe, mem_setOf_eq,
          forall_apply_eq_imp_iff₂, exists_imp, and_imp] at hx hy ⊢
        /-
          R : Type u
          L : Type v
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : LieAlgebra R L
          L₂ : Type w
          inst✝¹ : LieRing L₂
          inst✝ : LieAlgebra R L₂
          f : LieHom R L L₂
          K K' : LieSubalgebra R L
          K₂ : LieSubalgebra R L₂
          S : Set (LieSubalgebra R L)
          x y : L
          hx : ∀ (a : LieSubalgebra R L), Membership.mem S a → Membership.mem (↑a.toSubm …
          hy : ∀ (a : LieSubalgebra R L), Membership.mem S a → Membership.mem (↑a.toSubm …
          ⊢ ∀ (a : LieSubalgebra R L), Membership.mem S a → Membership.mem (↑a.toSubmodu …
        -/
        intro K hK
        /-
          R : Type u
          L : Type v
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : LieAlgebra R L
          L₂ : Type w
          inst✝¹ : LieRing L₂
          inst✝ : LieAlgebra R L₂
          f : LieHom R L L₂
          K✝ K' : LieSubalgebra R L
          K₂ : LieSubalgebra R L₂
          S : Set (LieSubalgebra R L)
          x y : L
          hx : ∀ (a : LieSubalgebra R L), Membership.mem S a → Membership.mem (↑a.toSubm …
          hy : ∀ (a : LieSubalgebra R L), Membership.mem S a → Membership.mem (↑a.toSubm …
          K : LieSubalgebra R L
          hK : Membership.mem S K
          ⊢ Membership.mem (↑K.toSubmodule) (Bracket.bracket x y)
        -/
        exact K.lie_mem (hx K hK) (hy K hK) }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem inf_coe : (↑(K ⊓ K') : Set L) = (K : Set L) ∩ (K' : Set L) :=
  rfl


@[simp]
theorem sInf_toSubmodule (S : Set (LieSubalgebra R L)) :
    (↑(sInf S) : Submodule R L) = sInf {(s : Submodule R L) | s ∈ S} :=
  rfl


@[deprecated (since := "2024-12-30")] alias sInf_coe_to_submodule := sInf_toSubmodule


@[simp]
theorem sInf_coe (S : Set (LieSubalgebra R L)) : (↑(sInf S) : Set L) = ⋂ s ∈ S, (s : Set L) := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    ⊢ Eq (↑(InfSet.sInf S)) (Set.iInter fun s => Set.iInter fun h => ↑s)
  -/
  rw [← coe_toSubmodule, sInf_toSubmodule, Submodule.sInf_coe]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    ⊢ Eq (Set.iInter fun p => Set.iInter fun h => ↑p) (Set.iInter fun s => Set.iIn …
  -/
  ext x
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    x : L
    ⊢ Iff (Membership.mem (Set.iInter fun p => Set.iInter fun h => ↑p) x) (Members …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem sInf_glb (S : Set (LieSubalgebra R L)) : IsGLB S (sInf S) := by
  have h : ∀ K K' : LieSubalgebra R L, (K : Set L) ≤ K' ↔ K ≤ K' := by
    intros
    exact Iff.rfl
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    h : ∀ (K K' : LieSubalgebra R L), Iff (LE.le ↑K ↑K') (LE.le K K')
    ⊢ IsGLB S (InfSet.sInf S)
  -/
  apply IsGLB.of_image @h
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    h : ∀ (K K' : LieSubalgebra R L), Iff (LE.le ↑K ↑K') (LE.le K K')
    ⊢ IsGLB (Set.image SetLike.coe S) ↑(InfSet.sInf S)
  -/
  simp only [sInf_coe]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    S : Set (LieSubalgebra R L)
    h : ∀ (K K' : LieSubalgebra R L), Iff (LE.le ↑K ↑K') (LE.le K K')
    ⊢ IsGLB (Set.image SetLike.coe S) (Set.iInter fun s => Set.iInter fun h => ↑s)
  -/
  exact isGLB_biInf
  /-
    🎉 no goals
  -/


/-- The set of Lie subalgebras of a Lie algebra form a complete lattice.

We provide explicit values for the fields `bot`, `top`, `inf` to get more convenient definitions
than we would otherwise obtain from `completeLatticeOfInf`. -/
instance completeLattice : CompleteLattice (LieSubalgebra R L) :=
  { completeLatticeOfInf _ sInf_glb with
    bot := ⊥
    bot_le := fun N _ h ↦ by
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        N : LieSubalgebra R L
        x✝ : L
        h : Membership.mem Bot.bot x✝
        ⊢ Membership.mem N x✝
      -/
      rw [mem_bot] at h
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        N : LieSubalgebra R L
        x✝ : L
        h : Eq x✝ 0
        ⊢ Membership.mem N x✝
      -/
      rw [h]
      /-
        R : Type u
        L : Type v
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        L₂ : Type w
        inst✝¹ : LieRing L₂
        inst✝ : LieAlgebra R L₂
        f : LieHom R L L₂
        K K' : LieSubalgebra R L
        K₂ : LieSubalgebra R L₂
        N : LieSubalgebra R L
        x✝ : L
        h : Eq x✝ 0
        ⊢ Membership.mem N 0
      -/
      exact N.zero_mem'
      /-
        🎉 no goals
      -/
    top := ⊤
    le_top := fun _ _ _ ↦ trivial
    inf := (· ⊓ ·)
    le_inf := fun _ _ _ h₁₂ h₁₃ _ hm ↦ ⟨h₁₂ hm, h₁₃ hm⟩
    inf_le_left := fun _ _ _ ↦ And.left
    inf_le_right := fun _ _ _ ↦ And.right }


instance : Add (LieSubalgebra R L) where add := max


instance : Zero (LieSubalgebra R L) where zero := ⊥


instance addCommMonoid : AddCommMonoid (LieSubalgebra R L) where
  add_assoc := sup_assoc
  zero_add := bot_sup_eq
  add_zero := sup_bot_eq
  add_comm := sup_comm
  nsmul := nsmulRec


instance : CanonicallyOrderedAddCommMonoid (LieSubalgebra R L) :=
  { LieSubalgebra.addCommMonoid,
    LieSubalgebra.completeLattice with
    add_le_add_left := fun _a _b ↦ sup_le_sup_left
    exists_add_of_le := @fun _a b h ↦ ⟨b, (sup_eq_right.2 h).symm⟩
    le_self_add := fun _a _b ↦ le_sup_left }


@[simp]
theorem add_eq_sup : K + K' = K ⊔ K' :=
  rfl


@[simp]
theorem inf_toSubmodule :
    (↑(K ⊓ K') : Submodule R L) = (K : Submodule R L) ⊓ (K' : Submodule R L) :=
  rfl


@[deprecated (since := "2024-12-30")] alias inf_coe_to_submodule := inf_toSubmodule


@[simp]
theorem mem_inf (x : L) : x ∈ K ⊓ K' ↔ x ∈ K ∧ x ∈ K' := by
  rw [← mem_toSubmodule, ← mem_toSubmodule, ← mem_toSubmodule, inf_toSubmodule,
    Submodule.mem_inf]


theorem eq_bot_iff : K = ⊥ ↔ ∀ x : L, x ∈ K → x = 0 := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    ⊢ Iff (Eq K Bot.bot) (∀ (x : L), Membership.mem K x → Eq x 0)
  -/
  rw [_root_.eq_bot_iff]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    ⊢ Iff (LE.le K Bot.bot) (∀ (x : L), Membership.mem K x → Eq x 0)
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


instance subsingleton_of_bot : Subsingleton (LieSubalgebra R (⊥ : LieSubalgebra R L)) := by
  /-
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    K K' : LieSubalgebra R L
    K₂ : LieSubalgebra R L₂
    ⊢ Subsingleton (LieSubalgebra R (Subtype fun x => Membership.mem Bot.bot x))
  -/
  apply subsingleton_of_bot_eq_top
  /-
    case hα
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    K K' : LieSubalgebra R L
    K₂ : LieSubalgebra R L₂
    ⊢ Eq Bot.bot Top.top
  -/
  ext ⟨x, hx⟩; change x ∈ ⊥ at hx; rw [LieSubalgebra.mem_bot] at hx; subst hx
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    K K' : LieSubalgebra R L
    K₂ : LieSubalgebra R L₂
    hx : Membership.mem Bot.bot 0
    ⊢ Iff (Membership.mem Bot.bot ⟨0, hx⟩) (Membership.mem Top.top ⟨0, hx⟩)
  -/
  simp only [mem_bot, mem_top, iff_true]
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    L₂ : Type w
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L L₂
    K K' : LieSubalgebra R L
    K₂ : LieSubalgebra R L₂
    hx : Membership.mem Bot.bot 0
    ⊢ Eq ⟨0, hx⟩ 0
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem subsingleton_bot : Subsingleton (⊥ : LieSubalgebra R L) :=
                                                         /-
                                                           R : Type u
                                                           L : Type v
                                                           inst✝² : CommRing R
                                                           inst✝¹ : LieRing L
                                                           inst✝ : LieAlgebra R L
                                                           ⊢ Subsingleton ↑↑Bot.bot
                                                         -/
  show Subsingleton ((⊥ : LieSubalgebra R L) : Set L) by simp
                                                         /-
                                                           🎉 no goals
                                                         -/


instance wellFoundedGT_of_noetherian [IsNoetherian R L] : WellFoundedGT (LieSubalgebra R L) :=
  RelHomClass.isWellFounded (⟨toSubmodule, @fun _ _ h ↦ h⟩ : _ →r (· > ·))


/-- Given two nested Lie subalgebras `K ⊆ K'`, the inclusion `K ↪ K'` is a morphism of Lie
algebras. -/
def inclusion : K →ₗ⁅R⁆ K' :=
  { Submodule.inclusion h with map_lie' := @fun _ _ ↦ rfl }


@[simp]
theorem coe_inclusion (x : K) : (inclusion h x : L) = x :=
  rfl


theorem inclusion_apply (x : K) : inclusion h x = ⟨x.1, h x.2⟩ :=
  rfl


theorem inclusion_injective : Function.Injective (inclusion h) := fun x y ↦ by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    x y : Subtype fun x => Membership.mem K x
    ⊢ Eq ((LieSubalgebra.inclusion h) x) ((LieSubalgebra.inclusion h) y) → Eq x y
  -/
  simp only [inclusion_apply, imp_self, Subtype.mk_eq_mk, SetLike.coe_eq_coe]
  /-
    🎉 no goals
  -/


/-- Given two nested Lie subalgebras `K ⊆ K'`, we can view `K` as a Lie subalgebra of `K'`,
regarded as Lie algebra in its own right. -/
def ofLe : LieSubalgebra R K' :=
  (inclusion h).range


@[simp]
theorem mem_ofLe (x : K') : x ∈ ofLe h ↔ (x : L) ∈ K := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    x : Subtype fun x => Membership.mem K' x
    ⊢ Iff (Membership.mem (LieSubalgebra.ofLe h) x) (Membership.mem K ↑x)
  -/
  simp only [ofLe, inclusion_apply, LieHom.mem_range]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    x : Subtype fun x => Membership.mem K' x
    ⊢ Iff (Exists fun y => Eq ⟨↑y, ⋯⟩ x) (Membership.mem K ↑x)
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h : LE.le K K'
      x : Subtype fun x => Membership.mem K' x
      ⊢ (Exists fun y => Eq ⟨↑y, ⋯⟩ x) → Membership.mem K ↑x
    -/
  · rintro ⟨y, rfl⟩
    /-
      case mp.intro
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h : LE.le K K'
      y : Subtype fun x => Membership.mem K x
      ⊢ Membership.mem K ↑⟨↑y, ⋯⟩
    -/
    exact y.property
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h : LE.le K K'
      x : Subtype fun x => Membership.mem K' x
      ⊢ Membership.mem K ↑x → Exists fun y => Eq ⟨↑y, ⋯⟩ x
    -/
  · intro h
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h✝ : LE.le K K'
      x : Subtype fun x => Membership.mem K' x
      h : Membership.mem K ↑x
      ⊢ Exists fun y => Eq ⟨↑y, ⋯⟩ x
    -/
    use ⟨(x : L), h⟩
    /-
      🎉 no goals
    -/


theorem ofLe_eq_comap_incl : ofLe h = K.comap K'.incl := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    ⊢ Eq (LieSubalgebra.ofLe h) (LieSubalgebra.comap K'.incl K)
  -/
  ext
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    x✝ : Subtype fun x => Membership.mem K' x
    ⊢ Iff (Membership.mem (LieSubalgebra.ofLe h) x✝) (Membership.mem (LieSubalgebr …
  -/
  rw [mem_ofLe]
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    x✝ : Subtype fun x => Membership.mem K' x
    ⊢ Iff (Membership.mem K ↑x✝) (Membership.mem (LieSubalgebra.comap K'.incl K) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_ofLe : (ofLe h : Submodule R K') = LinearMap.range (Submodule.inclusion h) :=
  rfl


/-- Given nested Lie subalgebras `K ⊆ K'`, there is a natural equivalence from `K` to its image in
`K'`. -/
noncomputable def equivOfLe : K ≃ₗ⁅R⁆ ofLe h :=
  (inclusion h).equivRangeOfInjective (inclusion_injective h)


@[simp]
theorem equivOfLe_apply (x : K) : equivOfLe h x = ⟨inclusion h x, (inclusion h).mem_range_self x⟩ :=
  rfl


theorem map_le_iff_le_comap {K : LieSubalgebra R L} {K' : LieSubalgebra R L₂} :
    map f K ≤ K' ↔ K ≤ comap f K' :=
  Set.image_subset_iff


theorem gc_map_comap : GaloisConnection (map f) (comap f) := fun _ _ ↦ map_le_iff_le_comap


/-- The Lie subalgebra of a Lie algebra `L` generated by a subset `s ⊆ L`. -/
def lieSpan : LieSubalgebra R L :=
  sInf { N | s ⊆ N }


theorem mem_lieSpan {x : L} : x ∈ lieSpan R L s ↔ ∀ K : LieSubalgebra R L, s ⊆ K → x ∈ K := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    x : L
    ⊢ Iff (Membership.mem (LieSubalgebra.lieSpan R L s) x) (∀ (K : LieSubalgebra R …
  -/
  change x ∈ (lieSpan R L s : Set L) ↔ _
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    x : L
    ⊢ Iff (Membership.mem (↑(LieSubalgebra.lieSpan R L s)) x) (∀ (K : LieSubalgebr …
  -/
  erw [sInf_coe]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    x : L
    ⊢ Iff (Membership.mem (Set.iInter fun s_1 => Set.iInter fun h => ↑s_1) x) (∀ ( …
  -/
  exact Set.mem_iInter₂
  /-
    🎉 no goals
  -/


theorem subset_lieSpan : s ⊆ lieSpan R L s := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    ⊢ HasSubset.Subset s ↑(LieSubalgebra.lieSpan R L s)
  -/
  intro m hm
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    m : L
    hm : Membership.mem s m
    ⊢ Membership.mem (↑(LieSubalgebra.lieSpan R L s)) m
  -/
  erw [mem_lieSpan]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    m : L
    hm : Membership.mem s m
    ⊢ ∀ (K : LieSubalgebra R L), HasSubset.Subset s ↑K → Membership.mem K m
  -/
  intro K hK
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    m : L
    hm : Membership.mem s m
    K : LieSubalgebra R L
    hK : HasSubset.Subset s ↑K
    ⊢ Membership.mem K m
  -/
  exact hK hm
  /-
    🎉 no goals
  -/


theorem submodule_span_le_lieSpan : Submodule.span R s ≤ lieSpan R L s := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    ⊢ LE.le (Submodule.span R s) (LieSubalgebra.lieSpan R L s).toSubmodule
  -/
  rw [Submodule.span_le]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    ⊢ HasSubset.Subset s ↑(LieSubalgebra.lieSpan R L s).toSubmodule
  -/
  apply subset_lieSpan
  /-
    🎉 no goals
  -/


theorem lieSpan_le {K} : lieSpan R L s ≤ K ↔ s ⊆ K := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s : Set L
    K : LieSubalgebra R L
    ⊢ Iff (LE.le (LieSubalgebra.lieSpan R L s) K) (HasSubset.Subset s ↑K)
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      s : Set L
      K : LieSubalgebra R L
      ⊢ LE.le (LieSubalgebra.lieSpan R L s) K → HasSubset.Subset s ↑K
    -/
  · exact Set.Subset.trans subset_lieSpan
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      s : Set L
      K : LieSubalgebra R L
      ⊢ HasSubset.Subset s ↑K → LE.le (LieSubalgebra.lieSpan R L s) K
    -/
  · intro hs m hm
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      s : Set L
      K : LieSubalgebra R L
      hs : HasSubset.Subset s ↑K
      m : L
      hm : Membership.mem (LieSubalgebra.lieSpan R L s) m
      ⊢ Membership.mem K m
    -/
    rw [mem_lieSpan] at hm
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      s : Set L
      K : LieSubalgebra R L
      hs : HasSubset.Subset s ↑K
      m : L
      hm : ∀ (K : LieSubalgebra R L), HasSubset.Subset s ↑K → Membership.mem K m
      ⊢ Membership.mem K m
    -/
    exact hm _ hs
    /-
      🎉 no goals
    -/


theorem lieSpan_mono {t : Set L} (h : s ⊆ t) : lieSpan R L s ≤ lieSpan R L t := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s t : Set L
    h : HasSubset.Subset s t
    ⊢ LE.le (LieSubalgebra.lieSpan R L s) (LieSubalgebra.lieSpan R L t)
  -/
  rw [lieSpan_le]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    s t : Set L
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset s ↑(LieSubalgebra.lieSpan R L t)
  -/
  exact Set.Subset.trans h subset_lieSpan
  /-
    🎉 no goals
  -/


theorem lieSpan_eq : lieSpan R L (K : Set L) = K :=
  le_antisymm (lieSpan_le.mpr rfl.subset) subset_lieSpan


theorem coe_lieSpan_submodule_eq_iff {p : Submodule R L} :
    (lieSpan R L (p : Set L) : Submodule R L) = p ↔ ∃ K : LieSubalgebra R L, ↑K = p := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    p : Submodule R L
    ⊢ Iff (Eq (LieSubalgebra.lieSpan R L ↑p).toSubmodule p) (Exists fun K => Eq K. …
  -/
  rw [p.exists_lieSubalgebra_coe_eq_iff]; constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      h : Eq (LieSubalgebra.lieSpan R L ↑p).toSubmodule p
      ⊢ ∀ (x y : L), Membership.mem p x → Membership.mem p y → Membership.mem p (Bra …
    -/
  · intro x m hm
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      h : Eq (LieSubalgebra.lieSpan R L ↑p).toSubmodule p
      x m : L
      hm : Membership.mem p x
      ⊢ Membership.mem p m → Membership.mem p (Bracket.bracket x m)
    -/
    rw [← h, mem_toSubmodule]
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      h : Eq (LieSubalgebra.lieSpan R L ↑p).toSubmodule p
      x m : L
      hm : Membership.mem p x
      ⊢ Membership.mem (LieSubalgebra.lieSpan R L ↑p) m → Membership.mem (LieSubalge …
    -/
    exact lie_mem _ (subset_lieSpan hm)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      p : Submodule R L
      h : ∀ (x y : L), Membership.mem p x → Membership.mem p y → Membership.mem p (B …
      ⊢ Eq (LieSubalgebra.lieSpan R L ↑p).toSubmodule p
    -/
  · rw [← toSubmodule_mk p @h, coe_toSubmodule, toSubmodule_inj, lieSpan_eq]
    /-
      🎉 no goals
    -/


/-- `lieSpan` forms a Galois insertion with the coercion from `LieSubalgebra` to `Set`. -/
protected def gi : GaloisInsertion (lieSpan R L : Set L → LieSubalgebra R L) (↑) where
  choice s _ := lieSpan R L s
  gc _ _ := lieSpan_le
  le_l_u _ := subset_lieSpan
  choice_eq _ _ := rfl


@[simp]
theorem span_empty : lieSpan R L (∅ : Set L) = ⊥ :=
  (LieSubalgebra.gi R L).gc.l_bot


@[simp]
theorem span_univ : lieSpan R L (Set.univ : Set L) = ⊤ :=
  eq_top_iff.2 <| SetLike.le_def.2 <| subset_lieSpan


theorem span_union (s t : Set L) : lieSpan R L (s ∪ t) = lieSpan R L s ⊔ lieSpan R L t :=
  (LieSubalgebra.gi R L).gc.l_sup


theorem span_iUnion {ι} (s : ι → Set L) : lieSpan R L (⋃ i, s i) = ⨆ i, lieSpan R L (s i) :=
  (LieSubalgebra.gi R L).gc.l_iSup


/-- If a predicate `p` is true on some set `s ⊆ L`, true for `0`, stable by scalar multiplication,
by addition and by Lie bracket, then the predicate is true on the Lie span of `s`. (Since `s` can be
empty, and the Lie span always contains `0`, the assumption that `p 0` holds cannot be removed.) -/
@[elab_as_elim]
theorem lieSpan_induction {p : L → Prop} {x : L} (h : x ∈ lieSpan R L s) (mem : ∀ x ∈ s, p x)
    (zero : p 0) (smul : ∀ (r : R), ∀ {x : L}, p x → p (r • x))
    (add : ∀ x y, p x → p y → p (x + y)) (lie : ∀ x y, p x → p y → p ⁅x, y⁆) : p x :=
  let S : LieSubalgebra R L :=
    { carrier := p
      add_mem' := add _ _
      zero_mem' := zero
      smul_mem' := smul
      lie_mem' := lie _ _ }
  lieSpan_le.mpr (show s ≤ S from mem) h


/-- An injective Lie algebra morphism is an equivalence onto its range. -/
noncomputable def ofInjective (f : L₁ →ₗ⁅R⁆ L₂) (h : Function.Injective f) : L₁ ≃ₗ⁅R⁆ f.range :=
                                                    /-
                                                      R : Type u
                                                      L₁ : Type v
                                                      L₂ : Type w
                                                      inst✝⁴ : CommRing R
                                                      inst✝³ : LieRing L₁
                                                      inst✝² : LieRing L₂
                                                      inst✝¹ : LieAlgebra R L₁
                                                      inst✝ : LieAlgebra R L₂
                                                      f : LieHom R L₁ L₂
                                                      h : Function.Injective ⇑f
                                                      ⊢ Function.Injective ⇑↑f
                                                    -/
  { LinearEquiv.ofInjective (f : L₁ →ₗ[R] L₂) <| by rwa [LieHom.coe_toLinearMap] with
                                                    /-
                                                      🎉 no goals
                                                    -/
    map_lie' := @fun x y ↦ SetCoe.ext <| f.map_lie x y }


@[simp]
theorem ofInjective_apply (f : L₁ →ₗ⁅R⁆ L₂) (h : Function.Injective f) (x : L₁) :
    ↑(ofInjective f h x) = f x :=
  rfl


/-- Lie subalgebras that are equal as sets are equivalent as Lie algebras. -/
def ofEq (h : (L₁' : Set L₁) = L₁'') : L₁' ≃ₗ⁅R⁆ L₁'' :=
  { LinearEquiv.ofEq (L₁' : Submodule R L₁) (L₁'' : Submodule R L₁) (by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        h : Eq ↑L₁' ↑L₁''
        ⊢ Eq L₁'.toSubmodule L₁''.toSubmodule
      -/
      ext x
      /-
        case h
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        h : Eq ↑L₁' ↑L₁''
        x : L₁
        ⊢ Iff (Membership.mem L₁'.toSubmodule x) (Membership.mem L₁''.toSubmodule x)
      -/
      change x ∈ (L₁' : Set L₁) ↔ x ∈ (L₁'' : Set L₁)
      /-
        case h
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        h : Eq ↑L₁' ↑L₁''
        x : L₁
        ⊢ Iff (Membership.mem (↑L₁') x) (Membership.mem (↑L₁'') x)
      -/
      rw [h]) with
      /-
        🎉 no goals
      -/
    map_lie' := @fun _ _ ↦ rfl }


@[simp]
theorem ofEq_apply (L L' : LieSubalgebra R L₁) (h : (L : Set L₁) = L') (x : L) :
    (↑(ofEq L L' h x) : L₁) = x :=
  rfl


/-- An equivalence of Lie algebras restricts to an equivalence from any Lie subalgebra onto its
image. -/
def lieSubalgebraMap : L₁'' ≃ₗ⁅R⁆ (L₁''.map e : LieSubalgebra R L₂) :=
  { LinearEquiv.submoduleMap (e : L₁ ≃ₗ[R] L₂) ↑L₁'' with
    map_lie' := @fun x y ↦ by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        x y : Subtype fun x => Membership.mem L₁'' x
        ⊢ Eq ((↑__src✝).toFun (Bracket.bracket x y)) (Bracket.bracket ((↑__src✝).toFun …
      -/
      apply SetCoe.ext
      /-
        case a
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        x y : Subtype fun x => Membership.mem L₁'' x
        ⊢ Eq ↑((↑__src✝).toFun (Bracket.bracket x y)) ↑(Bracket.bracket ((↑__src✝).toF …
      -/
      exact LieHom.map_lie (↑e : L₁ →ₗ⁅R⁆ L₂) ↑x ↑y }
      /-
        🎉 no goals
      -/


@[simp]
theorem lieSubalgebraMap_apply (x : L₁'') : ↑(e.lieSubalgebraMap _ x) = e x :=
  rfl


/-- An equivalence of Lie algebras restricts to an equivalence from any Lie subalgebra onto its
image. -/
def ofSubalgebras (h : L₁'.map ↑e = L₂') : L₁' ≃ₗ⁅R⁆ L₂' :=
  { LinearEquiv.ofSubmodules (e : L₁ ≃ₗ[R] L₂) (↑L₁') (↑L₂') (by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        h : Eq (LieSubalgebra.map e.toLieHom L₁') L₂'
        ⊢ Eq (Submodule.map (↑e.toLinearEquiv) L₁'.toSubmodule) L₂'.toSubmodule
      -/
      rw [← h]
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        h : Eq (LieSubalgebra.map e.toLieHom L₁') L₂'
        ⊢ Eq (Submodule.map (↑e.toLinearEquiv) L₁'.toSubmodule) (LieSubalgebra.map e.t …
      -/
      rfl) with
      /-
        🎉 no goals
      -/
    map_lie' := @fun x y ↦ by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        h : Eq (LieSubalgebra.map e.toLieHom L₁') L₂'
        x y : Subtype fun x => Membership.mem L₁' x
        ⊢ Eq ((↑__src✝).toFun (Bracket.bracket x y)) (Bracket.bracket ((↑__src✝).toFun …
      -/
      apply SetCoe.ext
      /-
        case a
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        L₁' L₁'' : LieSubalgebra R L₁
        L₂' : LieSubalgebra R L₂
        e : LieEquiv R L₁ L₂
        h : Eq (LieSubalgebra.map e.toLieHom L₁') L₂'
        x y : Subtype fun x => Membership.mem L₁' x
        ⊢ Eq ↑((↑__src✝).toFun (Bracket.bracket x y)) ↑(Bracket.bracket ((↑__src✝).toF …
      -/
      exact LieHom.map_lie (↑e : L₁ →ₗ⁅R⁆ L₂) ↑x ↑y }
      /-
        🎉 no goals
      -/


@[simp]
theorem ofSubalgebras_apply (h : L₁'.map ↑e = L₂') (x : L₁') : ↑(e.ofSubalgebras _ _ h x) = e x :=
  rfl


@[simp]
theorem ofSubalgebras_symm_apply (h : L₁'.map ↑e = L₂') (x : L₂') :
    ↑((e.ofSubalgebras _ _ h).symm x) = e.symm x :=
  rfl


