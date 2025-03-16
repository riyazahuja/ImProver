/--
The category of profinite groups. A term of this type consists of a profinite
set with a topological group structure.
-/
@[pp_with_univ]
structure ProfiniteGrp where
  /-- The underlying profinite topological space. -/
  toProfinite : Profinite
  /-- The group structure. -/
  [group : Group toProfinite]
  /-- The above data together form a topological group. -/
  [topologicalGroup : TopologicalGroup toProfinite]


/--
The category of profinite additive groups. A term of this type consists of a profinite
set with a topological additive group structure.
-/
@[pp_with_univ]
structure ProfiniteAddGrp where
  /-- The underlying profinite topological space. -/
  toProfinite : Profinite
  /-- The additive group structure. -/
  [addGroup : AddGroup toProfinite]
  /-- The above data together form a topological additive group. -/
  [topologicalAddGroup : TopologicalAddGroup toProfinite]


@[to_additive]
instance : CoeSort ProfiniteGrp (Type u) where
  coe G := G.toProfinite


@[to_additive]
instance : Category ProfiniteGrp where
  Hom A B := ContinuousMonoidHom A B
  id A := ContinuousMonoidHom.id A
  comp f g := ContinuousMonoidHom.comp g f


@[to_additive]
instance (G H : ProfiniteGrp) : FunLike (G ⟶ H) G H :=
  inferInstanceAs <| FunLike (ContinuousMonoidHom G H) G H


@[to_additive]
instance (G H : ProfiniteGrp) : MonoidHomClass (G ⟶ H) G H :=
  inferInstanceAs <| MonoidHomClass (ContinuousMonoidHom G H) G H


@[to_additive]
instance (G H : ProfiniteGrp) : ContinuousMapClass (G ⟶ H) G H :=
  inferInstanceAs <| ContinuousMapClass (ContinuousMonoidHom G H) G H


@[to_additive]
instance : ConcreteCategory ProfiniteGrp where
  forget :=
  { obj := fun G => G
    map := fun f => f }
  forget_faithful :=
    { map_injective := by
        /-
          ⊢ ∀ {X Y : ProfiniteGrp.{?u.12718}}, Function.Injective { obj := fun G => ↑G.t …
        -/
        intro G H f g h
        /-
          G H : ProfiniteGrp.{?u.12718}
          f g : Quiver.Hom G H
          h : Eq ({ obj := fun G => ↑G.toProfinite.toTop, map := fun {X Y} f => ⇑f, map_ …
          ⊢ Eq f g
        -/
        exact DFunLike.ext _ _ <| fun x => congr_fun h x }
        /-
          🎉 no goals
        -/


/-- Construct a term of `ProfiniteGrp` from a type endowed with the structure of a
compact and totally disconnected topological group.
(The condition of being Hausdorff can be omitted here because totally disconnected implies that {1}
is a closed set, thus implying Hausdorff in a topological group.)-/
@[to_additive "Construct a term of `ProfiniteAddGrp` from a type endowed with the structure of a
compact and totally disconnected topological additive group.
(The condition of being Hausdorff can be omitted here because totally disconnected implies that {0}
is a closed set, thus implying Hausdorff in a topological additive group.)"]
def of (G : Type u) [Group G] [TopologicalSpace G] [TopologicalGroup G]
    [CompactSpace G] [TotallyDisconnectedSpace G] : ProfiniteGrp where
  toProfinite := .of G
  group := ‹_›
  topologicalGroup := ‹_›


@[to_additive (attr := simp)]
theorem coe_of (X : ProfiniteGrp) : (of X : Type _) = X :=
  rfl


@[to_additive (attr := simp)]
theorem coe_id (X : ProfiniteGrp) : (𝟙 ((forget ProfiniteGrp).obj X)) = id :=
  rfl


@[to_additive (attr := simp)]
theorem coe_comp {X Y Z : ProfiniteGrp} (f : X ⟶ Y) (g : Y ⟶ Z) :
    ((forget ProfiniteGrp).map f ≫ (forget ProfiniteGrp).map g) = g ∘ f :=
  rfl


/-- Construct a term of `ProfiniteGrp` from a type endowed with the structure of a
profinite topological group. -/
@[to_additive "Construct a term of `ProfiniteAddGrp` from a type endowed with the structure of a
profinite topological additive group."]
abbrev ofProfinite (G : Profinite) [Group G] [TopologicalGroup G] :
    ProfiniteGrp := of G


/-- The pi-type of profinite groups is a profinite group. -/
@[to_additive "The pi-type of profinite additive groups is a
profinite additive group."]
def pi {α : Type u} (β : α → ProfiniteGrp) : ProfiniteGrp :=
  let pitype := Profinite.pi fun (a : α) => (β a).toProfinite
  letI (a : α): Group (β a).toProfinite := (β a).group
  letI : Group pitype := Pi.group
  letI : TopologicalGroup pitype := Pi.topologicalGroup
  ofProfinite pitype


/-- A `FiniteGrp` when given the discrete topology can be considered as a profinite group. -/
@[to_additive "A `FiniteAddGrp` when given the discrete topology can be considered as a
profinite additive group."]
def ofFiniteGrp (G : FiniteGrp) : ProfiniteGrp :=
  letI : TopologicalSpace G := ⊥
  letI : DiscreteTopology G := ⟨rfl⟩
  letI : TopologicalGroup G := {}
  of G


@[to_additive]
instance : HasForget₂ FiniteGrp ProfiniteGrp where
  forget₂ :=
  { obj := ofFiniteGrp
                           /-
                             X✝ Y✝ : FiniteGrp.{?u.23005}
                             f : Quiver.Hom X✝ Y✝
                             ⊢ Continuous (↑f).toFun
                           -/
    map := fun f => ⟨f, by continuity⟩ }
                           /-
                             🎉 no goals
                           -/


@[to_additive]
instance : HasForget₂ ProfiniteGrp Grp where
  forget₂ := {
    obj := fun P => ⟨P, P.group⟩
    map := fun f => f.toMonoidHom
  }


/-- A closed subgroup of a profinite group is profinite. -/
def ofClosedSubgroup {G : ProfiniteGrp} (H : ClosedSubgroup G)  : ProfiniteGrp :=
  letI : CompactSpace H := inferInstance
  of H.1


/-- A topological group that has a `ContinuousMulEquiv` to a profinite group is profinite. -/
def ofContinuousMulEquiv {G : ProfiniteGrp.{u}} {H : Type v} [TopologicalSpace H]
    [Group H] [TopologicalGroup H] (e : G ≃ₜ* H) : ProfiniteGrp.{v} :=
  let _ : CompactSpace H := Homeomorph.compactSpace e.toHomeomorph
  let _ : TotallyDisconnectedSpace H := Homeomorph.totallyDisconnectedSpace e.toHomeomorph
  .of H


/-- The functor mapping a profinite group to its underlying profinite space. -/
def profiniteGrpToProfinite : ProfiniteGrp ⥤ Profinite where
  obj G := G.toProfinite
                  /-
                    X✝ Y✝ : ProfiniteGrp.{?u.32206}
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Continuous ⇑f
                  -/
  map f := ⟨f, by continuity⟩
                  /-
                    🎉 no goals
                  -/


instance : profiniteGrpToProfinite.Faithful := {
  map_injective := fun {_ _} _ _ h =>
    ConcreteCategory.hom_ext_iff.mpr (congrFun (congrArg ContinuousMap.toFun h)) }


/-- Auxiliary construction to obtain the group structure on the limit of profinite groups. -/
def limitConePtAux : Subgroup (Π j : J, F.obj j) where
  carrier := {x | ∀ ⦃i j : J⦄ (π : i ⟶ j), F.map π (x i) = x j}
                             /-
                               J : Type v
                               inst✝ : CategoryTheory.SmallCategory J
                               F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
                               a✝ b✝ : (j : J) → ↑(F.obj j).toProfinite.toTop
                               hx : Membership.mem (setOf fun x => ∀ ⦃i j : J⦄ (π : Quiver.Hom i j), Eq ((F.m …
                               hy : Membership.mem (setOf fun x => ∀ ⦃i j : J⦄ (π : Quiver.Hom i j), Eq ((F.m …
                               x✝¹ x✝ : J
                               π : Quiver.Hom x✝¹ x✝
                               ⊢ Eq ((F.map π) (HMul.hMul a✝ b✝ x✝¹)) (HMul.hMul a✝ b✝ x✝)
                             -/
  mul_mem' hx hy _ _ π := by simp only [Pi.mul_apply, map_mul, hx π, hy π]
                             /-
                               🎉 no goals
                             -/
                 /-
                   J : Type v
                   inst✝ : CategoryTheory.SmallCategory J
                   F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
                   ⊢ Membership.mem { carrier := setOf fun x => ∀ ⦃i j : J⦄ (π : Quiver.Hom i j), …
                 -/
  one_mem' := by simp only [Set.mem_setOf_eq, Pi.one_apply, map_one, implies_true]
                 /-
                   🎉 no goals
                 -/
                         /-
                           J : Type v
                           inst✝ : CategoryTheory.SmallCategory J
                           F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
                           x✝² : (j : J) → ↑(F.obj j).toProfinite.toTop
                           h : Membership.mem { carrier := setOf fun x => ∀ ⦃i j : J⦄ (π : Quiver.Hom i j …
                           x✝¹ x✝ : J
                           π : Quiver.Hom x✝¹ x✝
                           ⊢ Eq ((F.map π) (Inv.inv x✝² x✝¹)) (Inv.inv x✝² x✝)
                         -/
  inv_mem' h _ _ π := by simp only [Pi.inv_apply, map_inv, h π]
                         /-
                           🎉 no goals
                         -/


instance : Group (Profinite.limitCone (F ⋙ profiniteGrpToProfinite.{max v u})).pt :=
  inferInstanceAs (Group (limitConePtAux F))


instance : TopologicalGroup (Profinite.limitCone (F ⋙ profiniteGrpToProfinite.{max v u})).pt :=
  inferInstanceAs (TopologicalGroup (limitConePtAux F))


/-- The explicit limit cone in `ProfiniteGrp`. -/
abbrev limitCone : Limits.Cone F where
  pt := ofProfinite (Profinite.limitCone (F ⋙ profiniteGrpToProfinite.{max v u})).pt
  π :=
  { app := fun j => {
      toFun := fun x => x.1 j
      map_one' := rfl
      map_mul' := fun x y => rfl
      continuous_toFun := by
        /-
          J : Type v
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
          j : J
          ⊢ Continuous (↑{ toFun := fun x => ↑x j, map_one' := ⋯, map_mul' := ⋯ }).toFun
        -/
        exact (continuous_apply j).comp (continuous_iff_le_induced.mpr fun U a => a) }
        /-
          🎉 no goals
        -/
    naturality := fun i j f => by
      simp only [Functor.const_obj_obj, Functor.comp_obj,
        Functor.const_obj_map, Category.id_comp, Functor.comp_map]
      /-
        J : Type v
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq { toFun := fun x => ↑x j, map_one' := ⋯, map_mul' := ⋯, continuous_toFun  …
      -/
      congr
      /-
        case e_toMonoidHom.e_toOneHom.e_toFun
        J : Type v
        inst✝ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
        i j : J
        f : Quiver.Hom i j
        ⊢ Eq (fun x => ↑x j) (Function.comp ⇑(F.map f).toMonoidHom ⇑{ toFun := fun x = …
      -/
      exact funext fun x => (x.2 f).symm }
      /-
        🎉 no goals
      -/


/-- `ProfiniteGrp.limitCone` is a limit cone. -/
def limitConeIsLimit : Limits.IsLimit (limitCone F) where
  lift cone := {
    ((Profinite.limitConeIsLimit (F ⋙ profiniteGrpToProfinite)).lift
      (profiniteGrpToProfinite.mapCone cone)) with
    map_one' := Subtype.ext (funext fun j ↦ map_one (cone.π.app j))
    -- TODO: investigate whether it's possible to set up `ext` lemmas for the `TopCat`-related
    -- categories so that `by ext j; exact map_one (cone.π.app j)` works here, similarly below.
    map_mul' := fun _ _ ↦ Subtype.ext (funext fun j ↦ map_mul (cone.π.app j) _ _) }
  uniq cone m h := by
    /-
      J : Type v
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ProfiniteGrp.{max v u}
      cone : CategoryTheory.Limits.Cone F
      m : Quiver.Hom cone.pt (ProfiniteGrp.limitCone F).pt
      h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((ProfiniteGrp.limitCo …
      ⊢ Eq m
          ((fun cone =>
              let __src := (Profinite.limitConeIsLimit (F.comp ProfiniteGrp.profinit …
              { toFun := __src.toFun, map_one' := ⋯, map_mul' := ⋯, continuous_toFun …
            cone)
    -/
    apply profiniteGrpToProfinite.map_injective
    simpa using (Profinite.limitConeIsLimit (F ⋙ profiniteGrpToProfinite)).uniq
      (profiniteGrpToProfinite.mapCone cone) (profiniteGrpToProfinite.map m)
      (fun j ↦ congrArg profiniteGrpToProfinite.map (h j))


instance : Limits.HasLimit F where
  exists_limit := Nonempty.intro
    { cone := limitCone F
      isLimit := limitConeIsLimit F }


/-- The abbreviation for the limit of `ProfiniteGrp`s. -/
abbrev limit : ProfiniteGrp := (ProfiniteGrp.limitCone F).pt


instance : Limits.PreservesLimits profiniteGrpToProfinite.{u} where
  preservesLimitsOfShape := {
    preservesLimit := fun {F} ↦ CategoryTheory.Limits.preservesLimit_of_preserves_limit_cone
      (limitConeIsLimit F) (Profinite.limitConeIsLimit (F ⋙ profiniteGrpToProfinite)) }


