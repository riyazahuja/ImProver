/-- The presheaf on `Profinite` of locally constant functions to `X`. -/
abbrev locallyConstantPresheaf : Profinite.{u}ᵒᵖ ⥤ Type (u+1) :=
  CompHausLike.LocallyConstant.functorToPresheaves.{u, u+1}.obj X


/--
The functor `locallyConstantPresheaf` takes cofiltered limits of finite sets with surjective
projection maps to colimits.
-/
noncomputable def isColimitLocallyConstantPresheaf (hc : IsLimit c) [∀ i, Epi (c.π.app i)] :
    IsColimit <| (locallyConstantPresheaf X).mapCocone c.op := by
  /-
    I : Type u
    inst✝² : CategoryTheory.Category.{u, u} I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    X : Type (u + 1)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    ⊢ CategoryTheory.Limits.IsColimit ((Condensed.locallyConstantPresheaf X).mapCo …
  -/
  refine Types.FilteredColimit.isColimitOf _ _ ?_ ?_
    /-
      case refine_1
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      ⊢ ∀ (x : ((Condensed.locallyConstantPresheaf X).mapCocone c.op).pt), Exists fu …
    -/
  · intro (f : LocallyConstant c.pt X)
    /-
      case refine_1
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      f : LocallyConstant (↑c.pt.toTop) X
      ⊢ Exists fun i => Exists fun xi => Eq f (((Condensed.locallyConstantPresheaf X …
    -/
    obtain ⟨j, h⟩ := Profinite.exists_locallyConstant.{_, u} c hc f
    /-
      case refine_1.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      f : LocallyConstant (↑c.pt.toTop) X
      j : I
      h : Exists fun g => Eq f (LocallyConstant.comap (c.π.app j) g)
      ⊢ Exists fun i => Exists fun xi => Eq f (((Condensed.locallyConstantPresheaf X …
    -/
    exact ⟨⟨j⟩, h⟩
    /-
      🎉 no goals
    -/
  · intro ⟨i⟩ ⟨j⟩ (fi : LocallyConstant _ _) (fj : LocallyConstant _ _)
      (h : fi.comap (c.π.app i) = fj.comap (c.π.app j))
    /-
      case refine_2
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq (((F.comp FintypeCat.toPr …
    -/
    obtain ⟨k, ki, kj, _⟩ := IsCofilteredOrEmpty.cone_objs i j
    /-
      case refine_2.intro.intro.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq (((F.comp FintypeCat.toPr …
    -/
    refine ⟨⟨k⟩, ki.op, kj.op, ?_⟩
    /-
      case refine_2.intro.intro.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Eq (((F.comp FintypeCat.toProfinite).op.comp (Condensed.locallyConstantPresh …
    -/
    dsimp
    /-
      case refine_2.intro.intro.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Eq (LocallyConstant.comap (FintypeCat.toProfinite.map (F.map ki)) fi) (Local …
    -/
    ext x
    /-
      case refine_2.intro.intro.intro.h
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x : ↑(FintypeCat.toProfinite.obj (F.obj k)).toTop
      ⊢ Eq ((LocallyConstant.comap (FintypeCat.toProfinite.map (F.map ki)) fi) x) (( …
    -/
    obtain ⟨x, hx⟩ := ((Profinite.epi_iff_surjective (c.π.app k)).mp inferInstance) x
    /-
      case refine_2.intro.intro.intro.h.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget Profinite).obj (((CategoryTheory.Functor.const I).o …
      hx : Eq ((c.π.app k) x) x✝
      ⊢ Eq ((LocallyConstant.comap (FintypeCat.toProfinite.map (F.map ki)) fi) x✝) ( …
    -/
    rw [← hx]
    change fi ((c.π.app k ≫ (F ⋙ toProfinite).map _) x) =
      fj ((c.π.app k ≫ (F ⋙ toProfinite).map _) x)
    /-
      case refine_2.intro.intro.intro.h.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget Profinite).obj (((CategoryTheory.Functor.const I).o …
      hx : Eq ((c.π.app k) x) x✝
      ⊢ Eq (fi ((CategoryTheory.CategoryStruct.comp (c.π.app k) ((F.comp FintypeCat. …
    -/
    have h := LocallyConstant.congr_fun h x
    /-
      case refine_2.intro.intro.intro.h.intro
      I : Type u
      inst✝² : CategoryTheory.Category.{u, u} I
      inst✝¹ : CategoryTheory.IsCofiltered I
      F : CategoryTheory.Functor I FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
      X : Type (u + 1)
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
      i j : I
      fi : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      fj : LocallyConstant (↑((F.comp FintypeCat.toProfinite).obj (Opposite.unop { u …
      h✝¹ : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.ap …
      k : I
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget Profinite).obj (((CategoryTheory.Functor.const I).o …
      hx : Eq ((c.π.app k) x) x✝
      h : Eq ((LocallyConstant.comap (c.π.app i) fi) x) ((LocallyConstant.comap (c.π …
      ⊢ Eq (fi ((CategoryTheory.CategoryStruct.comp (c.π.app k) ((F.comp FintypeCat. …
    -/
    rwa [c.w, c.w]
    /-
      🎉 no goals
    -/


@[simp]
lemma isColimitLocallyConstantPresheaf_desc_apply (hc : IsLimit c) [∀ i, Epi (c.π.app i)]
    (s : Cocone ((F ⋙ toProfinite).op ⋙ locallyConstantPresheaf X))
    (i : I) (f : LocallyConstant (toProfinite.obj (F.obj i)) X) :
    (isColimitLocallyConstantPresheaf c X hc).desc s (f.comap (c.π.app i)) = s.ι.app ⟨i⟩ f := by
  change ((((locallyConstantPresheaf X).mapCocone c.op).ι.app ⟨i⟩) ≫
    (isColimitLocallyConstantPresheaf c X hc).desc s) _ = _
  /-
    I : Type u
    inst✝² : CategoryTheory.Category.{u, u} I
    inst✝¹ : CategoryTheory.IsCofiltered I
    F : CategoryTheory.Functor I FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toProfinite)
    X : Type (u + 1)
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : I), CategoryTheory.Epi (c.π.app i)
    s : CategoryTheory.Limits.Cocone ((F.comp FintypeCat.toProfinite).op.comp (Con …
    i : I
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (F.obj i)).toTop) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((Condensed.locallyConstantPresheaf  …
  -/
  rw [(isColimitLocallyConstantPresheaf c X hc).fac]
  /-
    🎉 no goals
  -/


/-- `isColimitLocallyConstantPresheaf` in the case of `S.asLimit`. -/
noncomputable def isColimitLocallyConstantPresheafDiagram (S : Profinite) :
    IsColimit <| (locallyConstantPresheaf X).mapCocone S.asLimitCone.op :=
  isColimitLocallyConstantPresheaf _ _ S.asLimit


@[simp]
lemma isColimitLocallyConstantPresheafDiagram_desc_apply (S : Profinite)
    (s : Cocone (S.diagram.op ⋙ locallyConstantPresheaf X))
    (i : DiscreteQuotient S) (f : LocallyConstant (S.diagram.obj i) X) :
    (isColimitLocallyConstantPresheafDiagram X S).desc s (f.comap (S.asLimitCone.π.app i)) =
      s.ι.app ⟨i⟩ f :=
  isColimitLocallyConstantPresheaf_desc_apply S.asLimitCone X S.asLimit s i f


/--
Given a presheaf `F` on `Profinite`, `lanPresheaf F` is the left Kan extension of its
restriction to finite sets along the inclusion functor of finite sets into `Profinite`.
-/
abbrev lanPresheaf (F : Profinite.{u}ᵒᵖ ⥤ Type (u+1)) : Profinite.{u}ᵒᵖ ⥤ Type (u+1) :=
  pointwiseLeftKanExtension toProfinite.op (toProfinite.op ⋙ F)


/--
To presheaves on `Profinite` whose restrictions to finite sets are isomorphic have isomorphic left
Kan extensions.
-/
def lanPresheafExt {F G : Profinite.{u}ᵒᵖ ⥤ Type (u+1)}
    (i : toProfinite.op ⋙ F ≅ toProfinite.op ⋙ G) : lanPresheaf F ≅ lanPresheaf G :=
  leftKanExtensionUniqueOfIso _ (pointwiseLeftKanExtensionUnit _ _) i _
    (pointwiseLeftKanExtensionUnit _ _)


@[simp]
lemma lanPresheafExt_hom {F G : Profinite.{u}ᵒᵖ ⥤ Type (u+1)} (S : Profinite.{u}ᵒᵖ)
    (i : toProfinite.op ⋙ F ≅ toProfinite.op ⋙ G) : (lanPresheafExt i).hom.app S =
      colimMap (whiskerLeft (CostructuredArrow.proj toProfinite.op S) i.hom) := by
  simp only [lanPresheaf, pointwiseLeftKanExtension_obj, lanPresheafExt,
    leftKanExtensionUniqueOfIso_hom, pointwiseLeftKanExtension_desc_app]
  /-
    F G : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    S : Opposite Profinite
    i : CategoryTheory.Iso (FintypeCat.toProfinite.op.comp F) (FintypeCat.toProfin …
    ⊢ Eq (CategoryTheory.Limits.colimit.desc ((CategoryTheory.CostructuredArrow.pr …
  -/
  apply colimit.hom_ext
  /-
    case w
    F G : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    S : Opposite Profinite
    i : CategoryTheory.Iso (FintypeCat.toProfinite.op.comp F) (FintypeCat.toProfin …
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toProfinite.op S), Eq (Ca …
  -/
  aesop
  /-
    🎉 no goals
  -/


@[simp]
lemma lanPresheafExt_inv {F G : Profinite.{u}ᵒᵖ ⥤ Type (u+1)} (S : Profinite.{u}ᵒᵖ)
    (i : toProfinite.op ⋙ F ≅ toProfinite.op ⋙ G) : (lanPresheafExt i).inv.app S =
      colimMap (whiskerLeft (CostructuredArrow.proj toProfinite.op S) i.inv) := by
  simp only [lanPresheaf, pointwiseLeftKanExtension_obj, lanPresheafExt,
    leftKanExtensionUniqueOfIso_inv, pointwiseLeftKanExtension_desc_app]
  /-
    F G : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    S : Opposite Profinite
    i : CategoryTheory.Iso (FintypeCat.toProfinite.op.comp F) (FintypeCat.toProfin …
    ⊢ Eq (CategoryTheory.Limits.colimit.desc ((CategoryTheory.CostructuredArrow.pr …
  -/
  apply colimit.hom_ext
  /-
    case w
    F G : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    S : Opposite Profinite
    i : CategoryTheory.Iso (FintypeCat.toProfinite.op.comp F) (FintypeCat.toProfin …
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toProfinite.op S), Eq (Ca …
  -/
  aesop
  /-
    🎉 no goals
  -/


instance : Final <| Profinite.Extend.functorOp S.asLimitCone :=
  Profinite.Extend.functorOp_final S.asLimitCone S.asLimit


/--
A presheaf, which takes a profinite set written as a cofiltered limit to the corresponding
colimit, agrees with the left Kan extension of its restriction.
-/
def lanPresheafIso (hF : IsColimit <| F.mapCocone S.asLimitCone.op) :
    (lanPresheaf F).obj ⟨S⟩ ≅ F.obj ⟨S⟩ :=
  (Functor.Final.colimitIso (Profinite.Extend.functorOp S.asLimitCone) _).symm ≪≫
    (colimit.isColimit _).coconePointUniqueUpToIso hF


@[simp]
lemma lanPresheafIso_hom (hF : IsColimit <| F.mapCocone S.asLimitCone.op) :
    (lanPresheafIso hF).hom = colimit.desc _ (Profinite.Extend.cocone _ _) := by
  /-
    S : Profinite
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    hF : CategoryTheory.Limits.IsColimit (F.mapCocone S.asLimitCone.op)
    ⊢ Eq (Condensed.lanPresheafIso hF).hom (CategoryTheory.Limits.colimit.desc ((C …
  -/
  simp [lanPresheafIso, Final.colimitIso]
  /-
    S : Profinite
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    hF : CategoryTheory.Limits.IsColimit (F.mapCocone S.asLimitCone.op)
    ⊢ Eq ((CategoryTheory.Limits.colimit.isColimit (S.diagram.op.comp F)).coconePo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `lanPresheafIso` is natural in `S`. -/
def lanPresheafNatIso (hF : ∀ S : Profinite, IsColimit <| F.mapCocone S.asLimitCone.op) :
    lanPresheaf F ≅ F :=
  NatIso.ofComponents (fun ⟨S⟩ ↦ (lanPresheafIso (hF S)))
                /-
                  S : Profinite
                  F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
                  hF : (S : Profinite) → CategoryTheory.Limits.IsColimit (F.mapCocone S.asLimitC …
                  X✝ Y✝ : Opposite Profinite
                  x✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Condensed.lanPresheaf F).map x✝) (( …
                -/
    fun _ ↦ (by simpa using colimit.hom_ext fun _ ↦ (by simp))
                /-
                  🎉 no goals
                -/


@[simp]
lemma lanPresheafNatIso_hom_app (hF : ∀ S : Profinite, IsColimit <| F.mapCocone S.asLimitCone.op)
    (S : Profiniteᵒᵖ) : (lanPresheafNatIso hF).hom.app S =
      colimit.desc _ (Profinite.Extend.cocone _ _) := by
  /-
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    hF : (S : Profinite) → CategoryTheory.Limits.IsColimit (F.mapCocone S.asLimitC …
    S : Opposite Profinite
    ⊢ Eq ((Condensed.lanPresheafNatIso hF).hom.app S) (CategoryTheory.Limits.colim …
  -/
  simp [lanPresheafNatIso]
  /-
    🎉 no goals
  -/


/--
`lanPresheaf (locallyConstantPresheaf X)` is a sheaf for the coherent topology on `Profinite`.
-/
def lanSheafProfinite (X : Type (u+1)) : Sheaf (coherentTopology Profinite.{u}) (Type (u+1)) where
  val := lanPresheaf (locallyConstantPresheaf X)
  cond := by
    rw [Presheaf.isSheaf_of_iso_iff (lanPresheafNatIso
      fun _ ↦ isColimitLocallyConstantPresheafDiagram _ _)]
    exact ((CompHausLike.LocallyConstant.functor.{u, u+1}
      (hs := fun _ _ _ ↦ ((Profinite.effectiveEpi_tfae _).out 0 2).mp)).obj X).cond


/-- `lanPresheaf (locallyConstantPresheaf X)` as a condensed set. -/
def lanCondensedSet (X : Type (u+1)) : CondensedSet.{u} :=
  (ProfiniteCompHaus.equivalence _).functor.obj (lanSheafProfinite X)


/--
The functor which takes a finite set to the set of maps into `F(*)` for a presheaf `F` on
`Profinite`.
-/
@[simps]
def finYoneda : FintypeCat.{u}ᵒᵖ ⥤ Type (u+1) where
  obj X := X.unop → F.obj (toProfinite.op.obj ⟨of PUnit.{u+1}⟩)
  map f g := g ∘ f.unop


/-- `locallyConstantPresheaf` restricted to finite sets is isomorphic to `finYoneda F`. -/
@[simps! hom_app]
def locallyConstantIsoFinYoneda :
    toProfinite.op ⋙ (locallyConstantPresheaf (F.obj (toProfinite.op.obj ⟨of PUnit.{u+1}⟩))) ≅
    finYoneda F :=
  /-
    S : Profinite
    F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    ⊢ ∀ {X Y : Opposite FintypeCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categ …
  -/
  NatIso.ofComponents fun Y ↦ {
  /-
    🎉 no goals
  -/
    hom := fun f ↦ f.1
    inv := fun f ↦ ⟨f, @IsLocallyConstant.of_discrete _ _ _ ⟨rfl⟩ _⟩ }


/-- A finite set as a coproduct cocone in `Profinite` over itself. -/
def fintypeCatAsCofan (X : Profinite) :
    Cofan (fun (_ : X) ↦ (Profinite.of (PUnit.{u+1}))) :=
  Cofan.mk X (fun x ↦ (ContinuousMap.const _ x))


/-- A finite set is the coproduct of its points in `Profinite`. -/
def fintypeCatAsCofanIsColimit (X : Profinite) [Finite X] :
    IsColimit (fintypeCatAsCofan X) := by
  refine mkCofanColimit _ (fun t ↦ ⟨fun x ↦ t.inj x PUnit.unit, ?_⟩) ?_
    (fun _ _ h ↦ by ext x; exact ContinuousMap.congr_fun (h x) _)
    /-
      case refine_1
      S : Profinite
      F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
      X : Profinite
      inst✝ : Finite ↑X.toTop
      t : CategoryTheory.Limits.Cofan fun x => Profinite.of PUnit.{?u.129479 + 1}
      ⊢ Continuous fun x => (t.inj x) PUnit.unit
    -/
  · apply continuous_of_discreteTopology (α := X)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      S : Profinite
      F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
      X : Profinite
      inst✝ : Finite ↑X.toTop
      ⊢ ∀ (t : CategoryTheory.Limits.Cofan fun x => Profinite.of PUnit.{?u.129479 +  …
    -/
  · aesop
    /-
      🎉 no goals
    -/


noncomputable instance (X : Profinite) [Finite X] :
    PreservesLimitsOfShape (Discrete X) F :=
  let X' := (Countable.toSmall.{0} X).equiv_small.choose
  let e : X ≃ X' := (Countable.toSmall X).equiv_small.choose_spec.some
  have : Finite X' := .of_equiv X e
  preservesLimitsOfShape_of_equiv (Discrete.equivalence e.symm) F


/-- Auxiliary definition for `isoFinYoneda`. -/
def isoFinYonedaComponents (X : Profinite.{u}) [Finite X] :
    F.obj ⟨X⟩ ≅ (X → F.obj ⟨Profinite.of PUnit.{u+1}⟩) :=
  (isLimitFanMkObjOfIsLimit F _ _
    (Cofan.IsColimit.op (fintypeCatAsCofanIsColimit X))).conePointUniqueUpToIso
      (Types.productLimitCone.{u, u+1} fun _ ↦ F.obj ⟨Profinite.of PUnit.{u+1}⟩).2


lemma isoFinYonedaComponents_hom_apply (X : Profinite.{u}) [Finite X] (y : F.obj ⟨X⟩) (x : X) :
    (isoFinYonedaComponents F X).hom y x = F.map ((Profinite.of PUnit.{u+1}).const x).op y := rfl


lemma isoFinYonedaComponents_inv_comp {X Y : Profinite.{u}} [Finite X] [Finite Y]
    (f : Y → F.obj ⟨Profinite.of PUnit⟩) (g : X ⟶ Y) :
    (isoFinYonedaComponents F X).inv (f ∘ g) = F.map g.op ((isoFinYonedaComponents F Y).inv f) := by
  /-
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : Profinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := Profinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq ((Condensed.isoFinYonedaComponents F X).inv (Function.comp f ⇑g)) (F.map  …
  -/
  apply injective_of_mono (isoFinYonedaComponents F X).hom
  /-
    case a
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : Profinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := Profinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq ((Condensed.isoFinYonedaComponents F X).hom ((Condensed.isoFinYonedaCompo …
  -/
  simp only [CategoryTheory.inv_hom_id_apply]
  /-
    case a
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : Profinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := Profinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq (Function.comp f ⇑g) ((Condensed.isoFinYonedaComponents F X).hom (F.map g …
  -/
  ext x
  /-
    case a.h
    F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : Profinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := Profinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    x : ↑X.toTop
    ⊢ Eq (Function.comp f (⇑g) x) ((Condensed.isoFinYonedaComponents F X).hom (F.m …
  -/
  rw [isoFinYonedaComponents_hom_apply]
  simp only [← FunctorToTypes.map_comp_apply, ← op_comp, CompHausLike.const_comp,
    ← isoFinYonedaComponents_hom_apply, CategoryTheory.inv_hom_id_apply, Function.comp_apply]


/--
The restriction of a finite product preserving presheaf `F` on `Profinite` to the category of
finite sets is isomorphic to `finYoneda F`.
-/
@[simps!]
def isoFinYoneda : toProfinite.op ⋙ F ≅ finYoneda F :=
  NatIso.ofComponents (fun X ↦ isoFinYonedaComponents F (toProfinite.obj X.unop)) fun _ ↦ by
    /-
      S : Profinite
      F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((FintypeCat.toProfinite.op.comp F).m …
    -/
    simp only [comp_obj, op_obj, finYoneda_obj, Functor.comp_map, op_map]
    /-
      S : Profinite
      F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (FintypeCat.toProfinite.map x✝ …
    -/
    ext
    simp only [types_comp_apply, isoFinYonedaComponents_hom_apply, finYoneda_map,
      op_obj, Function.comp_apply, ← FunctorToTypes.map_comp_apply]
    /-
      case h.h
      S : Profinite
      F✝ F : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝¹ : Quiver.Hom X✝ Y✝
      a✝ : F.obj { unop := FintypeCat.toProfinite.obj (Opposite.unop X✝) }
      x✝ : ↑(Opposite.unop Y✝)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (FintypeCat.toProfinite.map x✝ …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
A presheaf `F`, which takes a profinite set written as a cofiltered limit to the corresponding
colimit, is isomorphic to the presheaf `LocallyConstant - F(*)`.
-/
def isoLocallyConstantOfIsColimit
    (hF : ∀ S : Profinite, IsColimit <| F.mapCocone S.asLimitCone.op) :
    F ≅ (locallyConstantPresheaf (F.obj (toProfinite.op.obj ⟨of PUnit.{u+1}⟩))) :=
  (lanPresheafNatIso hF).symm ≪≫
    lanPresheafExt (isoFinYoneda F ≪≫ (locallyConstantIsoFinYoneda F).symm) ≪≫
      lanPresheafNatIso fun _ ↦ isColimitLocallyConstantPresheafDiagram _ _


lemma isoLocallyConstantOfIsColimit_inv (X : Profinite.{u}ᵒᵖ ⥤ Type (u+1))
    [PreservesFiniteProducts X]
    (hX : ∀ S : Profinite.{u}, (IsColimit <| X.mapCocone S.asLimitCone.op)) :
    (isoLocallyConstantOfIsColimit X hX).inv =
      (CompHausLike.LocallyConstant.counitApp.{u, u+1} X) := by
  /-
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    ⊢ Eq (Condensed.isoLocallyConstantOfIsColimit X hX).inv (CompHausLike.LocallyC …
  -/
  dsimp [isoLocallyConstantOfIsColimit]
  /-
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Condensed.lanPresheafNatIso fun x => …
  -/
  rw [Iso.inv_comp_eq]
  /-
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Condensed.lanPresheafExt ((Condensed …
  -/
  ext S : 2
  /-
    case w.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Condensed.lanPresheafExt ((Condense …
  -/
  apply colimit.hom_ext
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toProfinite.op S), Eq (Ca …
  -/
  intro ⟨Y, _, g⟩
  simp? [locallyConstantIsoFinYoneda, isoFinYoneda, counitApp] says
    simp only [comp_obj, CostructuredArrow.proj_obj, op_obj, functorToPresheaves_obj_obj,
      isoFinYoneda, locallyConstantIsoFinYoneda, finYoneda_obj, LocallyConstant.toFun_eq_coe,
      NatTrans.comp_app, pointwiseLeftKanExtension_obj, lanPresheafExt_inv, Iso.trans_inv,
      Iso.symm_inv, whiskerLeft_comp, lanPresheafNatIso_hom_app, Opposite.op_unop, colimit.map_desc,
      id_eq, Functor.comp_map, op_map, colimit.ι_desc, Cocones.precompose_obj_pt,
      Profinite.Extend.cocone_pt, Cocones.precompose_obj_ι, Category.assoc, const_obj_obj,
      whiskerLeft_app, NatIso.ofComponents_hom_app, NatIso.ofComponents_inv_app,
      Profinite.Extend.cocone_ι_app, counitApp, colimit.ι_desc_assoc]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (CategoryTheory.Categor …
  -/
  erw [(counitApp.{u, u+1} X).naturality]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (CategoryTheory.Categor …
  -/
  simp only [← Category.assoc]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr
  /-
    case w.h.w.e_a
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (Condensed.isoFinYoneda …
  -/
  ext f
  /-
    case w.h.w.e_a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (Condensed.isoFinYoneda …
  -/
  simp only [types_comp_apply, isoFinYoneda_inv_app, counitApp_app]
  /-
    case w.h.w.e_a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    ⊢ Eq ((Condensed.isoFinYonedaComponents X (FintypeCat.toProfinite.obj (Opposit …
  -/
  apply presheaf_ext.{u, u+1} (X := X) (Y := X) (f := f)
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    ⊢ ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl …
  -/
  intro x
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((Condensed.isoFin …
  -/
  rw [incl_of_counitAppApp]
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((Condensed.isoFin …
  -/
  simp only [counitAppAppImage, CompHausLike.coe_of]
  letI : Fintype (fiber.{u, u+1} f x) :=
    Fintype.ofInjective (sigmaIncl.{u, u+1} f x).1 Subtype.val_injective
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((Condensed.isoFin …
  -/
  apply injective_of_mono (isoFinYonedaComponents X (fiber.{u, u+1} f x)).hom
  /-
    case w.h.w.e_a.h.h.a
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    ⊢ Eq ((Condensed.isoFinYonedaComponents X (CompHausLike.LocallyConstant.fiber  …
  -/
  ext y
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq ((Condensed.isoFinYonedaComponents X (CompHausLike.LocallyConstant.fiber  …
  -/
  simp only [isoFinYonedaComponents_hom_apply, ← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (CompHausLike.const (Profinite …
  -/
  rw [show (Profinite.of PUnit.{u+1}).const y ≫ IsTerminal.from _ (fiber f x) = 𝟙 _ from rfl]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (CompHausLike.const (Profinite …
  -/
  simp only [op_comp, FunctorToTypes.map_comp_apply, op_id, FunctorToTypes.map_id_apply]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CompHausLike.const (Profinite.of PUnit.{u + 1}) y).op (X.map (Com …
  -/
  rw [← isoFinYonedaComponents_inv_comp X _ (sigmaIncl.{u, u+1} f x)]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite Profinite) (Type (u + 1))
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : Profinite) → CategoryTheory.Limits.IsColimit (X.mapCocone S.asLimitC …
    S : Opposite Profinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toProfinite.op.obj Y) ((CategoryTheory.Functor.from …
    f : LocallyConstant (↑(FintypeCat.toProfinite.obj (Opposite.unop Y)).toTop) (X …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CompHausLike.const (Profinite.of PUnit.{u + 1}) y).op ((Condensed …
  -/
  simpa [← isoFinYonedaComponents_hom_apply] using x.map_eq_image f y
  /-
    🎉 no goals
  -/


/-- The presheaf on `LightProfinite` of locally constant functions to `X`. -/
abbrev locallyConstantPresheaf : LightProfiniteᵒᵖ ⥤ Type u :=
  CompHausLike.LocallyConstant.functorToPresheaves.{u, u}.obj X


/--
The functor `locallyConstantPresheaf` takes sequential limits of finite sets with surjective
projection maps to colimits.
-/
noncomputable def isColimitLocallyConstantPresheaf (hc : IsLimit c) [∀ i, Epi (c.π.app i)] :
    IsColimit <| (locallyConstantPresheaf X).mapCocone c.op := by
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    X : Type u
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    ⊢ CategoryTheory.Limits.IsColimit ((LightCondensed.locallyConstantPresheaf X). …
  -/
  refine Types.FilteredColimit.isColimitOf _ _ ?_ ?_
    /-
      case refine_1
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      ⊢ ∀ (x : ((LightCondensed.locallyConstantPresheaf X).mapCocone c.op).pt), Exis …
    -/
  · intro (f : LocallyConstant c.pt X)
    obtain ⟨j, h⟩ := Profinite.exists_locallyConstant.{_, 0} (lightToProfinite.mapCone c)
      (isLimitOfPreserves lightToProfinite hc) f
    /-
      case refine_1.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      f : LocallyConstant (↑c.pt.toTop) X
      j : Opposite Nat
      h : Exists fun g => Eq f (LocallyConstant.comap ((lightToProfinite.mapCone c). …
      ⊢ Exists fun i => Exists fun xi => Eq f (((LightCondensed.locallyConstantPresh …
    -/
    exact ⟨⟨j⟩, h⟩
    /-
      🎉 no goals
    -/
  · intro ⟨i⟩ ⟨j⟩ (fi : LocallyConstant _ _) (fj : LocallyConstant _ _)
      (h : fi.comap (c.π.app i) = fj.comap (c.π.app j))
    /-
      case refine_2
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq (((F.comp FintypeCat.toLi …
    -/
    obtain ⟨k, ki, kj, _⟩ := IsCofilteredOrEmpty.cone_objs i j
    /-
      case refine_2.intro.intro.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Exists fun k => Exists fun f => Exists fun g => Eq (((F.comp FintypeCat.toLi …
    -/
    refine ⟨⟨k⟩, ki.op, kj.op, ?_⟩
    /-
      case refine_2.intro.intro.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Eq (((F.comp FintypeCat.toLightProfinite).op.comp (LightCondensed.locallyCon …
    -/
    dsimp
    /-
      case refine_2.intro.intro.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      ⊢ Eq (LocallyConstant.comap (FintypeCat.toLightProfinite.map (F.map ki)) fi) ( …
    -/
    ext x
    /-
      case refine_2.intro.intro.intro.h
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x : ↑(FintypeCat.toLightProfinite.obj (F.obj k)).toTop
      ⊢ Eq ((LocallyConstant.comap (FintypeCat.toLightProfinite.map (F.map ki)) fi)  …
    -/
    obtain ⟨x, hx⟩ := ((LightProfinite.epi_iff_surjective (c.π.app k)).mp inferInstance) x
    /-
      case refine_2.intro.intro.intro.h.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toLightProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget LightProfinite).obj (((CategoryTheory.Functor.const …
      hx : Eq ((c.π.app k) x) x✝
      ⊢ Eq ((LocallyConstant.comap (FintypeCat.toLightProfinite.map (F.map ki)) fi)  …
    -/
    rw [← hx]
    change fi ((c.π.app k ≫ (F ⋙ toLightProfinite).map _) x) =
      fj ((c.π.app k ≫ (F ⋙ toLightProfinite).map _) x)
    /-
      case refine_2.intro.intro.intro.h.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.app  …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toLightProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget LightProfinite).obj (((CategoryTheory.Functor.const …
      hx : Eq ((c.π.app k) x) x✝
      ⊢ Eq (fi ((CategoryTheory.CategoryStruct.comp (c.π.app k) ((F.comp FintypeCat. …
    -/
    have h := LocallyConstant.congr_fun h x
    /-
      case refine_2.intro.intro.intro.h.intro
      F : CategoryTheory.Functor (Opposite Nat) FintypeCat
      c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
      X : Type u
      hc : CategoryTheory.Limits.IsLimit c
      inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
      i j : Opposite Nat
      fi : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      fj : LocallyConstant (↑((F.comp FintypeCat.toLightProfinite).obj (Opposite.uno …
      h✝¹ : Eq (LocallyConstant.comap (c.π.app i) fi) (LocallyConstant.comap (c.π.ap …
      k : Opposite Nat
      ki : Quiver.Hom k i
      kj : Quiver.Hom k j
      h✝ : True
      x✝ : ↑(FintypeCat.toLightProfinite.obj (F.obj k)).toTop
      x : (CategoryTheory.forget LightProfinite).obj (((CategoryTheory.Functor.const …
      hx : Eq ((c.π.app k) x) x✝
      h : Eq ((LocallyConstant.comap (c.π.app i) fi) x) ((LocallyConstant.comap (c.π …
      ⊢ Eq (fi ((CategoryTheory.CategoryStruct.comp (c.π.app k) ((F.comp FintypeCat. …
    -/
    rwa [c.w, c.w]
    /-
      🎉 no goals
    -/


@[simp]
lemma isColimitLocallyConstantPresheaf_desc_apply (hc : IsLimit c) [∀ i, Epi (c.π.app i)]
    (s : Cocone ((F ⋙ toLightProfinite).op ⋙ locallyConstantPresheaf X))
    (n : ℕᵒᵖ) (f : LocallyConstant (toLightProfinite.obj (F.obj n)) X) :
    (isColimitLocallyConstantPresheaf c X hc).desc s (f.comap (c.π.app n)) = s.ι.app ⟨n⟩ f := by
  change ((((locallyConstantPresheaf X).mapCocone c.op).ι.app ⟨n⟩) ≫
    (isColimitLocallyConstantPresheaf c X hc).desc s) _ = _
  /-
    F : CategoryTheory.Functor (Opposite Nat) FintypeCat
    c : CategoryTheory.Limits.Cone (F.comp FintypeCat.toLightProfinite)
    X : Type u
    hc : CategoryTheory.Limits.IsLimit c
    inst✝ : ∀ (i : Opposite Nat), CategoryTheory.Epi (c.π.app i)
    s : CategoryTheory.Limits.Cocone ((F.comp FintypeCat.toLightProfinite).op.comp …
    n : Opposite Nat
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (F.obj n)).toTop) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((LightCondensed.locallyConstantPres …
  -/
  rw [(isColimitLocallyConstantPresheaf c X hc).fac]
  /-
    🎉 no goals
  -/


/-- `isColimitLocallyConstantPresheaf` in the case of `S.asLimit`. -/
noncomputable def isColimitLocallyConstantPresheafDiagram (S : LightProfinite) :
    IsColimit <| (locallyConstantPresheaf X).mapCocone (coconeRightOpOfCone S.asLimitCone) :=
  (Functor.Final.isColimitWhiskerEquiv (opOpEquivalence ℕ).inverse _).symm
    (isColimitLocallyConstantPresheaf _ _ S.asLimit)


@[simp]
lemma isColimitLocallyConstantPresheafDiagram_desc_apply (S : LightProfinite)
    (s : Cocone (S.diagram.rightOp ⋙ locallyConstantPresheaf X))
    (n : ℕ) (f : LocallyConstant (S.diagram.obj ⟨n⟩) X) :
    (isColimitLocallyConstantPresheafDiagram X S).desc s (f.comap (S.asLimitCone.π.app ⟨n⟩)) =
      s.ι.app n f := by
  change ((((locallyConstantPresheaf X).mapCocone (coconeRightOpOfCone S.asLimitCone)).ι.app n) ≫
    (isColimitLocallyConstantPresheafDiagram X S).desc s) _ = _
  /-
    X : Type u
    S : LightProfinite
    s : CategoryTheory.Limits.Cocone (S.diagram.rightOp.comp (LightCondensed.local …
    n : Nat
    f : LocallyConstant (↑(S.diagram.obj { unop := n }).toTop) X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((LightCondensed.locallyConstantPres …
  -/
  rw [(isColimitLocallyConstantPresheafDiagram X S).fac]
  /-
    🎉 no goals
  -/


instance (S : LightProfinite.{u}ᵒᵖ) :
    HasColimitsOfShape (CostructuredArrow toLightProfinite.op S) (Type u) :=
  hasColimitsOfShape_of_equivalence (asEquivalence (CostructuredArrow.pre Skeleton.incl.op _ S))


/--
Given a presheaf `F` on `LightProfinite`, `lanPresheaf F` is the left Kan extension of its
restriction to finite sets along the inclusion functor of finite sets into `Profinite`.
-/
abbrev lanPresheaf (F : LightProfinite.{u}ᵒᵖ ⥤ Type u) : LightProfinite.{u}ᵒᵖ ⥤ Type u :=
  pointwiseLeftKanExtension toLightProfinite.op (toLightProfinite.op ⋙ F)


/--
To presheaves on `LightProfinite` whose restrictions to finite sets are isomorphic have isomorphic
left Kan extensions.
-/
def lanPresheafExt {F G : LightProfinite.{u}ᵒᵖ ⥤ Type u}
    (i : toLightProfinite.op ⋙ F ≅ toLightProfinite.op ⋙ G) : lanPresheaf F ≅ lanPresheaf G :=
  leftKanExtensionUniqueOfIso _ (pointwiseLeftKanExtensionUnit _ _) i _
    (pointwiseLeftKanExtensionUnit _ _)


@[simp]
lemma lanPresheafExt_hom {F G : LightProfinite.{u}ᵒᵖ ⥤ Type u} (S : LightProfinite.{u}ᵒᵖ)
    (i : toLightProfinite.op ⋙ F ≅ toLightProfinite.op ⋙ G) : (lanPresheafExt i).hom.app S =
      colimMap (whiskerLeft (CostructuredArrow.proj toLightProfinite.op S) i.hom) := by
  simp only [lanPresheaf, pointwiseLeftKanExtension_obj, lanPresheafExt,
    leftKanExtensionUniqueOfIso_hom, pointwiseLeftKanExtension_desc_app]
  /-
    F G : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    S : Opposite LightProfinite
    i : CategoryTheory.Iso (FintypeCat.toLightProfinite.op.comp F) (FintypeCat.toL …
    ⊢ Eq (CategoryTheory.Limits.colimit.desc ((CategoryTheory.CostructuredArrow.pr …
  -/
  apply colimit.hom_ext
  /-
    case w
    F G : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    S : Opposite LightProfinite
    i : CategoryTheory.Iso (FintypeCat.toLightProfinite.op.comp F) (FintypeCat.toL …
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toLightProfinite.op S), E …
  -/
  aesop
  /-
    🎉 no goals
  -/


@[simp]
lemma lanPresheafExt_inv  {F G : LightProfinite.{u}ᵒᵖ ⥤ Type u} (S : LightProfinite.{u}ᵒᵖ)
    (i : toLightProfinite.op ⋙ F ≅ toLightProfinite.op ⋙ G) : (lanPresheafExt i).inv.app S =
      colimMap (whiskerLeft (CostructuredArrow.proj toLightProfinite.op S) i.inv) := by
  simp only [lanPresheaf, pointwiseLeftKanExtension_obj, lanPresheafExt,
    leftKanExtensionUniqueOfIso_inv, pointwiseLeftKanExtension_desc_app]
  /-
    F G : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    S : Opposite LightProfinite
    i : CategoryTheory.Iso (FintypeCat.toLightProfinite.op.comp F) (FintypeCat.toL …
    ⊢ Eq (CategoryTheory.Limits.colimit.desc ((CategoryTheory.CostructuredArrow.pr …
  -/
  apply colimit.hom_ext
  /-
    case w
    F G : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    S : Opposite LightProfinite
    i : CategoryTheory.Iso (FintypeCat.toLightProfinite.op.comp F) (FintypeCat.toL …
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toLightProfinite.op S), E …
  -/
  aesop
  /-
    🎉 no goals
  -/


instance : Final <| LightProfinite.Extend.functorOp S.asLimitCone :=
  LightProfinite.Extend.functorOp_final S.asLimitCone S.asLimit


/--
A presheaf, which takes a light profinite set written as a sequential limit to the corresponding
colimit, agrees with the left Kan extension of its restriction.
-/
def lanPresheafIso (hF : IsColimit <| F.mapCocone (coconeRightOpOfCone S.asLimitCone)) :
    (lanPresheaf F).obj ⟨S⟩ ≅ F.obj ⟨S⟩ :=
  (Functor.Final.colimitIso (LightProfinite.Extend.functorOp S.asLimitCone) _).symm ≪≫
    (colimit.isColimit _).coconePointUniqueUpToIso hF


@[simp]
lemma lanPresheafIso_hom (hF : IsColimit <| F.mapCocone (coconeRightOpOfCone S.asLimitCone)) :
    (lanPresheafIso hF).hom = colimit.desc _ (LightProfinite.Extend.cocone _ _) := by
  /-
    S : LightProfinite
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    hF : CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.cocon …
    ⊢ Eq (LightCondensed.lanPresheafIso hF).hom (CategoryTheory.Limits.colimit.des …
  -/
  simp [lanPresheafIso, Final.colimitIso]
  /-
    S : LightProfinite
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    hF : CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.cocon …
    ⊢ Eq ((CategoryTheory.Limits.colimit.isColimit (S.diagram.rightOp.comp F)).coc …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `lanPresheafIso` is natural in `S`. -/
def lanPresheafNatIso
    (hF : ∀ S : LightProfinite, IsColimit <| F.mapCocone (coconeRightOpOfCone S.asLimitCone)) :
    lanPresheaf F ≅ F := by
  refine NatIso.ofComponents
    (fun ⟨S⟩ ↦ (lanPresheafIso (hF S))) fun _ ↦ ?_
  simp only [lanPresheaf, pointwiseLeftKanExtension_obj, pointwiseLeftKanExtension_map,
    lanPresheafIso_hom, Opposite.op_unop]
  /-
    S : LightProfinite
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    hF : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (F.mapCocone (Cate …
    X✝ Y✝ : Opposite LightProfinite
    x✝ : Quiver.Hom X✝ Y✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.desc ( …
  -/
  exact colimit.hom_ext fun _ ↦ (by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma lanPresheafNatIso_hom_app
    (hF : ∀ S : LightProfinite, IsColimit <| F.mapCocone (coconeRightOpOfCone S.asLimitCone))
    (S : LightProfiniteᵒᵖ) : (lanPresheafNatIso hF).hom.app S =
      colimit.desc _ (LightProfinite.Extend.cocone _ _) := by
  /-
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    hF : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (F.mapCocone (Cate …
    S : Opposite LightProfinite
    ⊢ Eq ((LightCondensed.lanPresheafNatIso hF).hom.app S) (CategoryTheory.Limits. …
  -/
  simp [lanPresheafNatIso]
  /-
    🎉 no goals
  -/


/--
`lanPresheaf (locallyConstantPresheaf X)` as a light condensed set.
-/
def lanLightCondSet (X : Type u) : LightCondSet.{u} where
  val := lanPresheaf (locallyConstantPresheaf X)
  cond := by
    rw [Presheaf.isSheaf_of_iso_iff (lanPresheafNatIso
      fun _ ↦ isColimitLocallyConstantPresheafDiagram _ _)]
    exact (CompHausLike.LocallyConstant.functor.{u, u}
      (hs := fun _ _ _ ↦ ((LightProfinite.effectiveEpi_iff_surjective _).mp)).obj X).cond


/--
The functor which takes a finite set to the set of maps into `F(*)` for a presheaf `F` on
`LightProfinite`.
-/
@[simps]
def finYoneda : FintypeCat.{u}ᵒᵖ ⥤ Type u where
  obj X := X.unop → F.obj (toLightProfinite.op.obj ⟨of PUnit.{u+1}⟩)
  map f g := g ∘ f.unop


/-- `locallyConstantPresheaf` restricted to finite sets is isomorphic to `finYoneda F`. -/
def locallyConstantIsoFinYoneda : toLightProfinite.op ⋙
    (locallyConstantPresheaf (F.obj (toLightProfinite.op.obj ⟨of PUnit.{u+1}⟩))) ≅ finYoneda F :=
  /-
    S : LightProfinite
    F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    ⊢ ∀ {X Y : Opposite FintypeCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categ …
  -/
  NatIso.ofComponents fun Y ↦ {
  /-
    🎉 no goals
  -/
    hom := fun f ↦ f.1
    inv := fun f ↦ ⟨f, @IsLocallyConstant.of_discrete _ _ _ ⟨rfl⟩ _⟩ }


/-- A finite set as a coproduct cocone in `LightProfinite` over itself. -/
def fintypeCatAsCofan (X : LightProfinite) :
    Cofan (fun (_ : X) ↦ (LightProfinite.of (PUnit.{u+1}))) :=
  Cofan.mk X (fun x ↦ (ContinuousMap.const _ x))


/-- A finite set is the coproduct of its points in `LightProfinite`. -/
def fintypeCatAsCofanIsColimit (X : LightProfinite) [Finite X] :
    IsColimit (fintypeCatAsCofan X) := by
  refine mkCofanColimit _ (fun t ↦ ⟨fun x ↦ t.inj x PUnit.unit, ?_⟩) ?_
    (fun _ _ h ↦ by ext x; exact ContinuousMap.congr_fun (h x) _)
    /-
      case refine_1
      S : LightProfinite
      F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
      X : LightProfinite
      inst✝ : Finite ↑X.toTop
      t : CategoryTheory.Limits.Cofan fun x => LightProfinite.of PUnit.{?u.634164 + 1}
      ⊢ Continuous fun x => (t.inj x) PUnit.unit
    -/
  · apply continuous_of_discreteTopology (α := X)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      S : LightProfinite
      F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
      X : LightProfinite
      inst✝ : Finite ↑X.toTop
      ⊢ ∀ (t : CategoryTheory.Limits.Cofan fun x => LightProfinite.of PUnit.{?u.6341 …
    -/
  · aesop
    /-
      🎉 no goals
    -/


noncomputable instance (X : FintypeCat.{u}) : PreservesLimitsOfShape (Discrete X) F :=
  let X' := (Countable.toSmall.{0} X).equiv_small.choose
  let e : X ≃ X' := (Countable.toSmall X).equiv_small.choose_spec.some
  have : Fintype X' := Fintype.ofEquiv X e
  preservesLimitsOfShape_of_equiv (Discrete.equivalence e.symm) F


/-- Auxiliary definition for `isoFinYoneda`. -/
def isoFinYonedaComponents (X : LightProfinite.{u}) [Finite X] :
    F.obj ⟨X⟩ ≅ (X → F.obj ⟨LightProfinite.of PUnit.{u+1}⟩) :=
  (isLimitFanMkObjOfIsLimit F _ _
    (Cofan.IsColimit.op (fintypeCatAsCofanIsColimit X))).conePointUniqueUpToIso
      (Types.productLimitCone.{u, u} fun _ ↦ F.obj ⟨LightProfinite.of PUnit.{u+1}⟩).2


lemma isoFinYonedaComponents_hom_apply (X : LightProfinite.{u}) [Finite X] (y : F.obj ⟨X⟩)
    (x : X) : (isoFinYonedaComponents F X).hom y x =
      F.map ((LightProfinite.of PUnit.{u+1}).const x).op y := rfl


lemma isoFinYonedaComponents_inv_comp {X Y : LightProfinite.{u}} [Finite X] [Finite Y]
    (f : Y → F.obj ⟨LightProfinite.of PUnit⟩) (g : X ⟶ Y) :
    (isoFinYonedaComponents F X).inv (f ∘ g) = F.map g.op ((isoFinYonedaComponents F Y).inv f) := by
  /-
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : LightProfinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := LightProfinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq ((LightCondensed.isoFinYonedaComponents F X).inv (Function.comp f ⇑g)) (F …
  -/
  apply injective_of_mono (isoFinYonedaComponents F X).hom
  /-
    case a
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : LightProfinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := LightProfinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq ((LightCondensed.isoFinYonedaComponents F X).hom ((LightCondensed.isoFinY …
  -/
  simp only [CategoryTheory.inv_hom_id_apply]
  /-
    case a
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : LightProfinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := LightProfinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    ⊢ Eq (Function.comp f ⇑g) ((LightCondensed.isoFinYonedaComponents F X).hom (F. …
  -/
  ext x
  /-
    case a.h
    F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝² : CategoryTheory.Limits.PreservesFiniteProducts F
    X Y : LightProfinite
    inst✝¹ : Finite ↑X.toTop
    inst✝ : Finite ↑Y.toTop
    f : ↑Y.toTop → F.obj { unop := LightProfinite.of PUnit.{u + 1} }
    g : Quiver.Hom X Y
    x : ↑X.toTop
    ⊢ Eq (Function.comp f (⇑g) x) ((LightCondensed.isoFinYonedaComponents F X).hom …
  -/
  rw [isoFinYonedaComponents_hom_apply]
  simp only [← FunctorToTypes.map_comp_apply, ← op_comp, CompHausLike.const_comp,
    ← isoFinYonedaComponents_hom_apply, CategoryTheory.inv_hom_id_apply, Function.comp_apply]


/--
The restriction of a finite product preserving presheaf `F` on `Profinite` to the category of
finite sets is isomorphic to `finYoneda F`.
-/
@[simps!]
def isoFinYoneda : toLightProfinite.op ⋙ F ≅ finYoneda F :=
  NatIso.ofComponents (fun X ↦ isoFinYonedaComponents F (toLightProfinite.obj X.unop)) fun _ ↦ by
    /-
      S : LightProfinite
      F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((FintypeCat.toLightProfinite.op.comp …
    -/
    simp only [comp_obj, op_obj, finYoneda_obj, Functor.comp_map, op_map]
    /-
      S : LightProfinite
      F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (FintypeCat.toLightProfinite.m …
    -/
    ext
    simp only [types_comp_apply, isoFinYonedaComponents_hom_apply, finYoneda_map, op_obj,
      Function.comp_apply, Types.productLimitCone, const_obj_obj, fintypeCatAsCofan, Cofan.mk_pt,
      cofan_mk_inj, Fan.mk_pt, Fan.mk_π_app, ← FunctorToTypes.map_comp_apply]
    /-
      case h.h
      S : LightProfinite
      F✝ F : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      X✝ Y✝ : Opposite FintypeCat
      x✝¹ : Quiver.Hom X✝ Y✝
      a✝ : F.obj { unop := FintypeCat.toLightProfinite.obj (Opposite.unop X✝) }
      x✝ : ↑(Opposite.unop Y✝)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (FintypeCat.toLightProfinite.m …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
A presheaf `F`, which takes a light profinite set written as a sequential limit to the corresponding
colimit, is isomorphic to the presheaf `LocallyConstant - F(*)`.
-/
def isoLocallyConstantOfIsColimit (hF : ∀ S : LightProfinite, IsColimit <|
    F.mapCocone (coconeRightOpOfCone S.asLimitCone)) :
      F ≅ (locallyConstantPresheaf
        (F.obj (toLightProfinite.op.obj ⟨of PUnit.{u+1}⟩))) :=
  (lanPresheafNatIso hF).symm ≪≫
    lanPresheafExt (isoFinYoneda F ≪≫ (locallyConstantIsoFinYoneda F).symm) ≪≫
      lanPresheafNatIso fun _ ↦ isColimitLocallyConstantPresheafDiagram _ _


lemma isoLocallyConstantOfIsColimit_inv (X : LightProfinite.{u}ᵒᵖ ⥤ Type u)
    [PreservesFiniteProducts X] (hX : ∀ S : LightProfinite.{u}, (IsColimit <|
      X.mapCocone (coconeRightOpOfCone S.asLimitCone))) :
    (isoLocallyConstantOfIsColimit X hX).inv =
      (CompHausLike.LocallyConstant.counitApp.{u, u} X) := by
  /-
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    ⊢ Eq (LightCondensed.isoLocallyConstantOfIsColimit X hX).inv (CompHausLike.Loc …
  -/
  dsimp [isoLocallyConstantOfIsColimit]
  /-
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (LightCondensed.lanPresheafNatIso fun …
  -/
  rw [Iso.inv_comp_eq]
  /-
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (LightCondensed.lanPresheafExt ((Ligh …
  -/
  ext S : 2
  /-
    case w.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (LightCondensed.lanPresheafExt ((Lig …
  -/
  apply colimit.hom_ext
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    ⊢ ∀ (j : CategoryTheory.CostructuredArrow FintypeCat.toLightProfinite.op S), E …
  -/
  intro ⟨Y, _, g⟩
  simp? [locallyConstantIsoFinYoneda, isoFinYoneda, counitApp] says
    simp only [comp_obj, CostructuredArrow.proj_obj, op_obj, functorToPresheaves_obj_obj,
      isoFinYoneda, locallyConstantIsoFinYoneda, finYoneda_obj, LocallyConstant.toFun_eq_coe,
      NatTrans.comp_app, pointwiseLeftKanExtension_obj, lanPresheafExt_inv, Iso.trans_inv,
      Iso.symm_inv, whiskerLeft_comp, lanPresheafNatIso_hom_app, Opposite.op_unop, colimit.map_desc,
      id_eq, Functor.comp_map, op_map, colimit.ι_desc, Cocones.precompose_obj_pt,
      LightProfinite.Extend.cocone_pt, Cocones.precompose_obj_ι, Category.assoc, const_obj_obj,
      whiskerLeft_app, NatIso.ofComponents_hom_app, NatIso.ofComponents_inv_app,
      LightProfinite.Extend.cocone_ι_app, counitApp, colimit.ι_desc_assoc]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (CategoryTheory.Categor …
  -/
  erw [(counitApp.{u, u} X).naturality]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (CategoryTheory.Categor …
  -/
  simp only [← Category.assoc]
  /-
    case w.h.w
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr
  /-
    case w.h.w.e_a
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (LightCondensed.isoFinY …
  -/
  ext f
  /-
    case w.h.w.e_a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => ⇑f) (LightCondensed.isoFinY …
  -/
  simp only [types_comp_apply, isoFinYoneda_inv_app, counitApp_app]
  /-
    case w.h.w.e_a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    ⊢ Eq ((LightCondensed.isoFinYonedaComponents X (FintypeCat.toLightProfinite.ob …
  -/
  apply presheaf_ext.{u, u} (X := X) (Y := X) (f := f)
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    ⊢ ∀ (a : Function.Fiber ⇑f), Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl …
  -/
  intro x
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((LightCondensed.i …
  -/
  rw [incl_of_counitAppApp]
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((LightCondensed.i …
  -/
  simp only [counitAppAppImage, CompHausLike.coe_of]
  letI : Fintype (fiber.{u, u} f x) :=
    Fintype.ofInjective (sigmaIncl.{u, u} f x).1 Subtype.val_injective
  /-
    case w.h.w.e_a.h.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    ⊢ Eq (X.map (CompHausLike.LocallyConstant.sigmaIncl f x).op ((LightCondensed.i …
  -/
  apply injective_of_mono (isoFinYonedaComponents X (fiber.{u, u} f x)).hom
  /-
    case w.h.w.e_a.h.h.a
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    ⊢ Eq ((LightCondensed.isoFinYonedaComponents X (CompHausLike.LocallyConstant.f …
  -/
  ext y
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq ((LightCondensed.isoFinYonedaComponents X (CompHausLike.LocallyConstant.f …
  -/
  simp only [isoFinYonedaComponents_hom_apply, ← FunctorToTypes.map_comp_apply, ← op_comp]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (CompHausLike.const (LightProf …
  -/
  rw [show (LightProfinite.of PUnit.{u+1}).const y ≫ IsTerminal.from _ (fiber f x) = 𝟙 _ from rfl]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp (CompHausLike.const (LightProf …
  -/
  simp only [op_comp, FunctorToTypes.map_comp_apply, op_id, FunctorToTypes.map_id_apply]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CompHausLike.const (LightProfinite.of PUnit.{u + 1}) y).op (X.map …
  -/
  rw [← isoFinYonedaComponents_inv_comp X _ (sigmaIncl.{u, u} f x)]
  /-
    case w.h.w.e_a.h.h.a.h
    X : CategoryTheory.Functor (Opposite LightProfinite) (Type u)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts X
    hX : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.mapCocone (Cate …
    S : Opposite LightProfinite
    Y : Opposite FintypeCat
    right✝ : CategoryTheory.Discrete PUnit.{1}
    g : Quiver.Hom (FintypeCat.toLightProfinite.op.obj Y) ((CategoryTheory.Functor …
    f : LocallyConstant (↑(FintypeCat.toLightProfinite.obj (Opposite.unop Y)).toTo …
    x : Function.Fiber ⇑f
    this : Fintype ↑(CompHausLike.LocallyConstant.fiber f x).toTop := Fintype.ofIn …
    y : ↑(CompHausLike.LocallyConstant.fiber f x).toTop
    ⊢ Eq (X.map (CompHausLike.const (LightProfinite.of PUnit.{u + 1}) y).op ((Ligh …
  -/
  simpa [← isoFinYonedaComponents_hom_apply] using x.map_eq_image f y
  /-
    🎉 no goals
  -/


