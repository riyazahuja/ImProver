/-- `AddGroupExtension N E G` is a short exact sequence of additive groups `0 → N → E → G → 0`. -/
structure AddGroupExtension [AddGroup N] [AddGroup E] [AddGroup G] where
  /-- The inclusion homomorphism `N →+ E` -/
  inl : N →+ E
  /-- The projection homomorphism `E →+ G` -/
  rightHom : E →+ G
  /-- The inclusion map is injective. -/
  inl_injective : Function.Injective inl
  /-- The range of the inclusion map is equal to the kernel of the projection map. -/
  range_inl_eq_ker_rightHom : inl.range = rightHom.ker
  /-- The projection map is surjective. -/
  rightHom_surjective : Function.Surjective rightHom


/-- `GroupExtension N E G` is a short exact sequence of groups `1 → N → E → G → 1`. -/
@[to_additive]
structure GroupExtension [Group N] [Group E] [Group G] where
  /-- The inclusion homomorphism `N →* E` -/
  inl : N →* E
  /-- The projection homomorphism `E →* G` -/
  rightHom : E →* G
  /-- The inclusion map is injective. -/
  inl_injective : Function.Injective inl
  /-- The range of the inclusion map is equal to the kernel of the projection map. -/
  range_inl_eq_ker_rightHom : inl.range = rightHom.ker
  /-- The projection map is surjective. -/
  rightHom_surjective : Function.Surjective rightHom


/-- `AddGroupExtension`s are equivalent iff there is a homomorphism making a commuting diagram. -/
structure Equiv {E' : Type*} [AddGroup E'] (S' : AddGroupExtension N E' G) where
  /-- The homomorphism -/
  toAddMonoidHom : E →+ E'
  /-- The left-hand side of the diagram commutes. -/
  inl_comm : toAddMonoidHom.comp S.inl = S'.inl
  /-- The right-hand side of the diagram commutes. -/
  rightHom_comm : S'.rightHom.comp toAddMonoidHom = S.rightHom


/-- `Splitting` of an additive group extension is a section homomorphism. -/
structure Splitting where
  /-- A section homomorphism -/
  sectionHom : G →+ E
  /-- The section is a left inverse of the projection map. -/
  rightHom_comp_sectionHom : S.rightHom.comp sectionHom = AddMonoidHom.id G


/-- The range of the inclusion map is a normal subgroup. -/
@[to_additive "The range of the inclusion map is a normal additive subgroup." ]
instance normal_inl_range : S.inl.range.Normal :=
  S.range_inl_eq_ker_rightHom ▸ S.rightHom.normal_ker


@[to_additive (attr := simp)]
theorem rightHom_inl (n : N) : S.rightHom (S.inl n) = 1 := by
  /-
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    n : N
    ⊢ Eq (S.rightHom (S.inl n)) 1
  -/
  rw [← MonoidHom.mem_ker, ← S.range_inl_eq_ker_rightHom, MonoidHom.mem_range]
  /-
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    n : N
    ⊢ Exists fun x => Eq (S.inl x) (S.inl n)
  -/
  exact exists_apply_eq_apply S.inl n
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem rightHom_comp_inl : S.rightHom.comp S.inl = 1 := by
  /-
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    ⊢ Eq (S.rightHom.comp S.inl) 1
  -/
  ext n
  /-
    case h
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    n : N
    ⊢ Eq ((S.rightHom.comp S.inl) n) (1 n)
  -/
  rw [MonoidHom.one_apply, MonoidHom.comp_apply]
  /-
    case h
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    n : N
    ⊢ Eq (S.rightHom (S.inl n)) 1
  -/
  exact S.rightHom_inl n
  /-
    🎉 no goals
  -/


/-- `E` acts on `N` by conjugation. -/
noncomputable def conjAct : E →* MulAut N where
  toFun e := (MonoidHom.ofInjective S.inl_injective).trans <|
    (MulAut.conjNormal e).trans (MonoidHom.ofInjective S.inl_injective).symm
  map_one' := by
    /-
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      ⊢ Eq ((fun e => (MonoidHom.ofInjective ⋯).trans (MulEquiv.trans (MulAut.conjNo …
    -/
    ext _
    /-
      case h
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      x✝ : N
      ⊢ Eq (((fun e => (MonoidHom.ofInjective ⋯).trans (MulEquiv.trans (MulAut.conjN …
    -/
    simp only [map_one, MulEquiv.trans_apply, MulAut.one_apply, MulEquiv.symm_apply_apply]
    /-
      🎉 no goals
    -/
  map_mul' _ _ := by
    /-
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      x✝¹ x✝ : E
      ⊢ Eq ({ toFun := fun e => (MonoidHom.ofInjective ⋯).trans (MulEquiv.trans (Mul …
    -/
    ext _
    /-
      case h
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      x✝² x✝¹ : E
      x✝ : N
      ⊢ Eq (({ toFun := fun e => (MonoidHom.ofInjective ⋯).trans (MulEquiv.trans (Mu …
    -/
    simp only [map_mul, MulEquiv.trans_apply, MulAut.mul_apply, MulEquiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


/-- The inclusion and a conjugation commute. -/
theorem inl_conjAct_comm {e : E} {n : N} : S.inl (S.conjAct e n) = e * S.inl n * e⁻¹ := by
  simp only [conjAct, MonoidHom.coe_mk, OneHom.coe_mk, MulEquiv.trans_apply,
    MonoidHom.apply_ofInjective_symm]
  /-
    N : Type u_1
    E : Type u_2
    G : Type u_3
    inst✝² : Group N
    inst✝¹ : Group E
    inst✝ : Group G
    S : GroupExtension N E G
    e : E
    n : N
    ⊢ Eq (↑((MulAut.conjNormal e) ((MonoidHom.ofInjective ⋯) n))) (HMul.hMul (HMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `GroupExtension`s are equivalent iff there is a homomorphism making a commuting diagram. -/
@[to_additive]
structure Equiv {E' : Type*} [Group E'] (S' : GroupExtension N E' G) where
  /-- The homomorphism -/
  toMonoidHom : E →* E'
  /-- The left-hand side of the diagram commutes. -/
  inl_comm : toMonoidHom.comp S.inl = S'.inl
  /-- The right-hand side of the diagram commutes. -/
  rightHom_comm : S'.rightHom.comp toMonoidHom = S.rightHom


/-- `Splitting` of a group extension is a section homomorphism. -/
@[to_additive]
structure Splitting where
  /-- A section homomorphism -/
  sectionHom : G →* E
  /-- The section is a left inverse of the projection map. -/
  rightHom_comp_sectionHom : S.rightHom.comp sectionHom = MonoidHom.id G


@[to_additive]
instance : FunLike S.Splitting G E where
  coe s := s.sectionHom
  coe_injective' := by
    /-
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      ⊢ Function.Injective fun s => ⇑s.sectionHom
    -/
    intro ⟨_, _⟩ ⟨_, _⟩ h
    /-
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      sectionHom✝¹ : MonoidHom G E
      rightHom_comp_sectionHom✝¹ : Eq (S.rightHom.comp sectionHom✝¹) (MonoidHom.id G)
      sectionHom✝ : MonoidHom G E
      rightHom_comp_sectionHom✝ : Eq (S.rightHom.comp sectionHom✝) (MonoidHom.id G)
      h : Eq ((fun s => ⇑s.sectionHom) { sectionHom := sectionHom✝¹, rightHom_comp_s …
      ⊢ Eq { sectionHom := sectionHom✝¹, rightHom_comp_sectionHom := rightHom_comp_s …
    -/
    congr
    /-
      case e_sectionHom
      N : Type u_1
      E : Type u_2
      G : Type u_3
      inst✝² : Group N
      inst✝¹ : Group E
      inst✝ : Group G
      S : GroupExtension N E G
      sectionHom✝¹ : MonoidHom G E
      rightHom_comp_sectionHom✝¹ : Eq (S.rightHom.comp sectionHom✝¹) (MonoidHom.id G)
      sectionHom✝ : MonoidHom G E
      rightHom_comp_sectionHom✝ : Eq (S.rightHom.comp sectionHom✝) (MonoidHom.id G)
      h : Eq ((fun s => ⇑s.sectionHom) { sectionHom := sectionHom✝¹, rightHom_comp_s …
      ⊢ Eq sectionHom✝¹ sectionHom✝
    -/
    exact DFunLike.coe_injective h
    /-
      🎉 no goals
    -/


@[to_additive]
instance : MonoidHomClass S.Splitting G E where
  map_mul s := s.sectionHom.map_mul'
  map_one s := s.sectionHom.map_one'


/-- A splitting of an extension `S` is `N`-conjugate to another iff there exists `n : N` such that
the section homomorphism is a conjugate of the other section homomorphism by `S.inl n`. -/
@[to_additive
      "A splitting of an extension `S` is `N`-conjugate to another iff there exists `n : N` such
      that the section homomorphism is a conjugate of the other section homomorphism by `S.inl n`."]
def IsConj (S : GroupExtension N E G) (s s' : S.Splitting) : Prop :=
  ∃ n : N, s.sectionHom = fun g ↦ S.inl n * s'.sectionHom g * (S.inl n)⁻¹


/-- The group extension associated to the semidirect product -/
def toGroupExtension : GroupExtension N (N ⋊[φ] G) G where
  inl_injective := inl_injective
  range_inl_eq_ker_rightHom := range_inl_eq_ker_rightHom
  rightHom_surjective := rightHom_surjective


theorem toGroupExtension_inl : (toGroupExtension φ).inl = SemidirectProduct.inl := rfl


theorem toGroupExtension_rightHom : (toGroupExtension φ).rightHom = SemidirectProduct.rightHom :=
  rfl


/-- A canonical splitting of the group extension associated to the semidirect product -/
def inr_splitting : (toGroupExtension φ).Splitting where
  sectionHom := inr
  rightHom_comp_sectionHom := rightHom_comp_inr


