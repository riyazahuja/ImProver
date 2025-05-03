/-- The category of finite groups. -/
@[pp_with_univ]
structure FiniteGrp where
  /-- A group that is finite -/
  toGrp : Grp
  [isFinite : Finite toGrp]


/-- The category of finite additive groups. -/
@[pp_with_univ]
structure FiniteAddGrp where
  /-- An add group that is finite -/
  toAddGrp : AddGrp
  [isFinite : Finite toAddGrp]


@[to_additive]
instance : CoeSort FiniteGrp.{u} (Type u) where
  coe G := G.toGrp


@[to_additive]
instance : Category FiniteGrp := InducedCategory.category FiniteGrp.toGrp


@[to_additive]
instance : ConcreteCategory FiniteGrp := InducedCategory.concreteCategory FiniteGrp.toGrp


@[to_additive]
instance (G : FiniteGrp) : Group G := inferInstanceAs <| Group G.toGrp


@[to_additive]
instance (G : FiniteGrp) : Finite G := G.isFinite


@[to_additive]
instance (G H : FiniteGrp) : FunLike (G ⟶ H) G H :=
  inferInstanceAs <| FunLike (G →* H) G H


@[to_additive]
instance (G H : FiniteGrp) : MonoidHomClass (G ⟶ H) G H :=
  inferInstanceAs <| MonoidHomClass (G →* H) G H


/-- Construct a term of `FiniteGrp` from a type endowed with the structure of a finite group. -/
@[to_additive "Construct a term of `FiniteAddGrp` from a type endowed with the structure of a
finite additive group."]
def of (G : Type u) [Group G] [Finite G] : FiniteGrp where
  toGrp := Grp.of G
  isFinite := ‹_›


/-- The morphism in `FiniteGrp`, induced from a morphism of the category `Grp`. -/
@[to_additive "The morphism in `FiniteAddGrp`, induced from a morphism of the category `AddGrp`"]
def ofHom {X Y : Type u} [Group X] [Finite X] [Group Y] [Finite Y] (f : X →* Y) : of X ⟶ of Y :=
  Grp.ofHom f


@[to_additive]
lemma ofHom_apply {X Y : Type u} [Group X] [Finite X] [Group Y] [Finite Y] (f : X →* Y) (x : X) :
    ofHom f x = f x :=
  rfl


