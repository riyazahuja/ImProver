/-- The colimit of `F ⋙ forget₂ Grp MonCat` in the category `MonCat`.
In the following, we will show that this has the structure of a group.
-/
@[to_additive
  "The colimit of `F ⋙ forget₂ AddGrp AddMonCat` in the category `AddMonCat`.
  In the following, we will show that this has the structure of an additive group."]
noncomputable abbrev G : MonCat :=
  MonCat.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ Grp MonCat.{max v u})


/-- The canonical projection into the colimit, as a quotient type. -/
@[to_additive "The canonical projection into the colimit, as a quotient type."]
abbrev G.mk : (Σ j, F.obj j) → G.{v, u} F :=
  Quot.mk (Types.Quot.Rel (F ⋙ forget Grp.{max v u}))


@[to_additive]
theorem G.mk_eq (x y : Σ j, F.obj j)
    (h : ∃ (k : J) (f : x.1 ⟶ k) (g : y.1 ⟶ k), F.map f x.2 = F.map g y.2) :
    G.mk.{v, u} F x = G.mk F y :=
  Quot.eqvGen_sound (Types.FilteredColimit.eqvGen_quot_rel_of_rel (F ⋙ forget Grp) x y h)


/-- The "unlifted" version of taking inverses in the colimit. -/
@[to_additive "The \"unlifted\" version of negation in the colimit."]
def colimitInvAux (x : Σ j, F.obj j) : G.{v, u} F :=
  G.mk F ⟨x.1, x.2⁻¹⟩


@[to_additive]
theorem colimitInvAux_eq_of_rel (x y : Σ j, F.obj j)
    (h : Types.FilteredColimit.Rel (F ⋙ forget Grp) x y) :
    colimitInvAux.{v, u} F x = colimitInvAux F y := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J Grp
    x y : Sigma fun j => ↑(F.obj j)
    h : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.fo …
    ⊢ Eq (Grp.FilteredColimits.colimitInvAux F x) (Grp.FilteredColimits.colimitInv …
  -/
  apply G.mk_eq
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J Grp
    x y : Sigma fun j => ↑(F.obj j)
    h : CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.fo …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) ⟨x.fst, Inv.in …
  -/
  obtain ⟨k, f, g, hfg⟩ := h
  /-
    case h.intro.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J Grp
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.comp (CategoryTheory.forget Grp)).map f x.snd) ((F.comp (Category …
    ⊢ Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) ⟨x.fst, Inv.in …
  -/
  use k, f, g
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J Grp
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.comp (CategoryTheory.forget Grp)).map f x.snd) ((F.comp (Category …
    ⊢ Eq ((F.map f) ⟨x.fst, Inv.inv x.snd⟩.snd) ((F.map g) ⟨y.fst, Inv.inv y.snd⟩. …
  -/
  rw [MonoidHom.map_inv, MonoidHom.map_inv, inv_inj]
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    F : CategoryTheory.Functor J Grp
    x y : Sigma fun j => ↑(F.obj j)
    k : J
    f : Quiver.Hom x.fst k
    g : Quiver.Hom y.fst k
    hfg : Eq ((F.comp (CategoryTheory.forget Grp)).map f x.snd) ((F.comp (Category …
    ⊢ Eq ((F.map f) x.snd) ((F.map g) y.snd)
  -/
  exact hfg
  /-
    🎉 no goals
  -/


/-- Taking inverses in the colimit. See also `colimitInvAux`. -/
@[to_additive "Negation in the colimit. See also `colimitNegAux`."]
instance colimitInv : Inv (G.{v, u} F) where
  inv x := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J Grp
      x : ↑(Grp.FilteredColimits.G F)
      ⊢ ↑(Grp.FilteredColimits.G F)
    -/
    refine Quot.lift (colimitInvAux.{v, u} F) ?_ x
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J Grp
      x : ↑(Grp.FilteredColimits.G F)
      ⊢ ∀ (a b : Sigma fun j => ↑(F.obj j)), CategoryTheory.Limits.Types.Quot.Rel (( …
    -/
    intro x y h
    /-
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J Grp
      x✝ : ↑(Grp.FilteredColimits.G F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((F.comp (CategoryTheory.forget₂ Grp  …
      ⊢ Eq (Grp.FilteredColimits.colimitInvAux F x) (Grp.FilteredColimits.colimitInv …
    -/
    apply colimitInvAux_eq_of_rel
    /-
      case h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J Grp
      x✝ : ↑(Grp.FilteredColimits.G F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((F.comp (CategoryTheory.forget₂ Grp  …
      ⊢ CategoryTheory.Limits.Types.FilteredColimit.Rel (F.comp (CategoryTheory.forg …
    -/
    apply Types.FilteredColimit.rel_of_quot_rel
    /-
      case h.a
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      F : CategoryTheory.Functor J Grp
      x✝ : ↑(Grp.FilteredColimits.G F)
      x y : Sigma fun j => ↑(F.obj j)
      h : CategoryTheory.Limits.Types.Quot.Rel ((F.comp (CategoryTheory.forget₂ Grp  …
      ⊢ CategoryTheory.Limits.Types.Quot.Rel (F.comp (CategoryTheory.forget Grp)) x y
    -/
    exact h
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem colimit_inv_mk_eq (x : Σ j, F.obj j) : (G.mk.{v, u} F x)⁻¹ = G.mk F ⟨x.1, x.2⁻¹⟩ :=
  rfl


@[to_additive]
noncomputable instance colimitGroup : Group (G.{v, u} F) :=
  { colimitInv.{v, u} F, (G.{v, u} F).str with
    inv_mul_cancel := fun x => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J Grp
        x : ↑(Grp.FilteredColimits.G F)
        ⊢ Eq (HMul.hMul (Inv.inv x) x) 1
      -/
      refine Quot.inductionOn x ?_; clear x; intro x
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J Grp
        x : Sigma fun j => ((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp (Categor …
        ⊢ Eq (HMul.hMul (Inv.inv (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.co …
      -/
      obtain ⟨j, x⟩ := x
      erw [colimit_inv_mk_eq,
        colimit_mul_mk_eq (F ⋙ forget₂ Grp MonCat.{max v u}) ⟨j, _⟩ ⟨j, _⟩ j (𝟙 j) (𝟙 j),
        colimit_one_eq (F ⋙ forget₂ Grp MonCat.{max v u}) j]
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J Grp
        j : J
        x : ((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp (CategoryTheory.forget  …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ Grp MonCat) …
      -/
      dsimp
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        inst✝ : CategoryTheory.IsFiltered J
        F : CategoryTheory.Functor J Grp
        j : J
        x : ((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp (CategoryTheory.forget  …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ Grp MonCat) …
      -/
      erw [CategoryTheory.Functor.map_id, inv_mul_cancel] }
      /-
        🎉 no goals
      -/


/-- The bundled group giving the filtered colimit of a diagram. -/
@[to_additive "The bundled additive group giving the filtered colimit of a diagram."]
noncomputable def colimit : Grp.{max v u} :=
  Grp.of (G.{v, u} F)


/-- The cocone over the proposed colimit group. -/
@[to_additive "The cocone over the proposed colimit additive group."]
noncomputable def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι := { (MonCat.FilteredColimits.colimitCocone (F ⋙ forget₂ Grp MonCat.{max v u})).ι with }


/-- The proposed colimit cocone is a colimit in `Grp`. -/
@[to_additive "The proposed colimit cocone is a colimit in `AddGroup`."]
def colimitCoconeIsColimit : IsColimit (colimitCocone.{v, u} F) where
  desc t :=
    MonCat.FilteredColimits.colimitDesc.{v, u} (F ⋙ forget₂ Grp MonCat.{max v u})
      ((forget₂ Grp MonCat).mapCocone t)
  fac t j :=
    DFunLike.coe_injective <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget Grp)).fac
      ((forget Grp).mapCocone t) j
  uniq t _ h :=
    DFunLike.coe_injective' <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget Grp)).uniq
      ((forget Grp).mapCocone t) _
        fun j => funext fun x => DFunLike.congr_fun (h j) x


@[to_additive forget₂AddMon_preservesFilteredColimits]
noncomputable instance forget₂Mon_preservesFilteredColimits :
    PreservesFilteredColimits.{u} (forget₂ Grp.{u} MonCat.{u}) where
      preserves_filtered_colimits x hx1 _ :=
      letI : Category.{u, u} x := hx1
      ⟨fun {F} => preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (MonCat.FilteredColimits.colimitCoconeIsColimit.{u, u} _)⟩


@[to_additive]
noncomputable instance forget_preservesFilteredColimits :
    PreservesFilteredColimits (forget Grp.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ Grp MonCat) (forget MonCat.{u})


/-- The colimit of `F ⋙ forget₂ CommGrp Grp` in the category `Grp`.
In the following, we will show that this has the structure of a _commutative_ group.
-/
@[to_additive
  "The colimit of `F ⋙ forget₂ AddCommGrp AddGrp` in the category `AddGrp`.
  In the following, we will show that this has the structure of a _commutative_ additive group."]
noncomputable abbrev G : Grp.{max v u} :=
  Grp.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ CommGrp.{max v u} Grp.{max v u})


@[to_additive]
noncomputable instance colimitCommGroup : CommGroup.{max v u} (G.{v, u} F) :=
  { (G F).str,
    CommMonCat.FilteredColimits.colimitCommMonoid
      (F ⋙ forget₂ CommGrp CommMonCat.{max v u}) with }


/-- The bundled commutative group giving the filtered colimit of a diagram. -/
@[to_additive "The bundled additive commutative group giving the filtered colimit of a diagram."]
noncomputable def colimit : CommGrp :=
  CommGrp.of (G.{v, u} F)


/-- The cocone over the proposed colimit commutative group. -/
@[to_additive "The cocone over the proposed colimit additive commutative group."]
noncomputable def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι :=
    { (Grp.FilteredColimits.colimitCocone
          (F ⋙ forget₂ CommGrp Grp.{max v u})).ι with }


/-- The proposed colimit cocone is a colimit in `CommGrp`. -/
@[to_additive "The proposed colimit cocone is a colimit in `AddCommGroup`."]
def colimitCoconeIsColimit : IsColimit (colimitCocone.{v, u} F) where
  desc t :=
    (Grp.FilteredColimits.colimitCoconeIsColimit.{v, u}
          (F ⋙ forget₂ CommGrp Grp.{max v u})).desc
      ((forget₂ CommGrp Grp.{max v u}).mapCocone t)
  fac t j :=
    DFunLike.coe_injective <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommGrp)).fac
        ((forget CommGrp).mapCocone t) j
  uniq t _ h :=
    DFunLike.coe_injective <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommGrp)).uniq
        ((forget CommGrp).mapCocone t) _ fun j => funext fun x => DFunLike.congr_fun (h j) x


@[to_additive]
noncomputable instance forget₂Group_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ CommGrp Grp.{u}) where
  preserves_filtered_colimits J hJ1 _ :=
    letI : Category J := hJ1
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (Grp.FilteredColimits.colimitCoconeIsColimit.{u, u}
            (F ⋙ forget₂ CommGrp Grp.{u})) }


@[to_additive]
noncomputable instance forget_preservesFilteredColimits :
    PreservesFilteredColimits (forget CommGrp.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ CommGrp Grp) (forget Grp.{u})


