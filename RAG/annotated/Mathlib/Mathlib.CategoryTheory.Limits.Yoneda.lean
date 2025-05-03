/-- The colimit cocone over `coyoneda.obj X`, with cocone point `PUnit`.
-/
@[simps]
def colimitCocone (X : Cᵒᵖ) : Cocone (coyoneda.obj X) where
  pt := PUnit
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X : Opposite C
                     ⊢ (X_1 : C) → Quiver.Hom ((CategoryTheory.coyoneda.obj X).obj X_1) (((Category …
                   -/
  ι := { app := by aesop_cat }
                   /-
                     🎉 no goals
                   -/


/-- The proposed colimit cocone over `coyoneda.obj X` is a colimit cocone.
-/
@[simps]
def colimitCoconeIsColimit (X : Cᵒᵖ) : IsColimit (colimitCocone X) where
  desc s _ := s.ι.app (unop X) (𝟙 _)
  fac s Y := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoneda.colimitCoco …
    -/
    funext f
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      Y : C
      f : (CategoryTheory.coyoneda.obj X).obj Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoneda.colimitCoco …
    -/
    convert congr_fun (s.w f).symm (𝟙 (unop X))
    simp only [coyoneda_obj_obj, Functor.const_obj_obj, types_comp_apply,
      coyoneda_obj_map, Category.id_comp]
  uniq s m w := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      m : Quiver.Hom (CategoryTheory.Coyoneda.colimitCocone X).pt s.pt
      w : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoned …
      ⊢ Eq m ((fun s x => s.ι.app (Opposite.unop X) (CategoryTheory.CategoryStruct.i …
    -/
    apply funext; rintro ⟨⟩
    /-
      case h.unit
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      m : Quiver.Hom (CategoryTheory.Coyoneda.colimitCocone X).pt s.pt
      w : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoned …
      ⊢ Eq (m PUnit.unit) ((fun s x => s.ι.app (Opposite.unop X) (CategoryTheory.Cat …
    -/
    dsimp
    /-
      case h.unit
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      m : Quiver.Hom (CategoryTheory.Coyoneda.colimitCocone X).pt s.pt
      w : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoned …
      ⊢ Eq (m PUnit.unit) (s.ι.app (Opposite.unop X) (CategoryTheory.CategoryStruct. …
    -/
    rw [← w]
    /-
      case h.unit
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X : Opposite C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.coyoneda.obj X)
      m : Quiver.Hom (CategoryTheory.Coyoneda.colimitCocone X).pt s.pt
      w : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyoned …
      ⊢ Eq (m PUnit.unit) (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Coyon …
    -/
    simp
    /-
      🎉 no goals
    -/


instance (X : Cᵒᵖ) : HasColimit (coyoneda.obj X) :=
  HasColimit.mk
    { cocone := _
      isColimit := colimitCoconeIsColimit X }


/-- The colimit of `coyoneda.obj X` is isomorphic to `PUnit`.
-/
noncomputable def colimitCoyonedaIso (X : Cᵒᵖ) : colimit (coyoneda.obj X) ≅ PUnit := by
  apply colimit.isoColimitCocone
    { cocone := _
      isColimit := colimitCoconeIsColimit X }


/-- The cone of `F` corresponding to an element in `(F ⋙ yoneda.obj X).sections`. -/
@[simps]
def Limits.coneOfSectionCompYoneda (F : J ⥤ Cᵒᵖ) (X : C)
    (s : (F ⋙ yoneda.obj X).sections) : Cone F where
  pt := Opposite.op X
  π := compYonedaSectionsEquiv F X s


instance yoneda_preservesLimit (F : J ⥤ Cᵒᵖ) (X : C) :
    PreservesLimit F (yoneda.obj X) where
  preserves {c} hc := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      X : C
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).mapCo …
    -/
    rw [Types.isLimit_iff]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      X : C
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (s : (j : J) → (F.comp (CategoryTheory.yoneda.obj X)).obj j), Membership.m …
    -/
    intro s hs
    exact ⟨(hc.lift (Limits.coneOfSectionCompYoneda F X ⟨s, hs⟩)).unop,
      fun j => Quiver.Hom.op_inj (hc.fac (Limits.coneOfSectionCompYoneda F X ⟨s, hs⟩) j),
      fun m hm => Quiver.Hom.op_inj
        (hc.uniq (Limits.coneOfSectionCompYoneda F X ⟨s, hs⟩) _
          (fun j => Quiver.Hom.unop_inj (hm j)))⟩


variable (J) in
noncomputable instance yoneda_preservesLimitsOfShape (X : C) :
    PreservesLimitsOfShape J (yoneda.obj X) where


/-- The yoneda embeddings jointly reflect limits. -/
def yonedaJointlyReflectsLimits (F : J ⥤ Cᵒᵖ) (c : Cone F)
    (hc : ∀ X : C, IsLimit ((yoneda.obj X).mapCone c)) : IsLimit c where
  lift s := ((hc s.pt.unop).lift ((yoneda.obj s.pt.unop).mapCone s) (𝟙 _)).op
  fac s j := Quiver.Hom.unop_inj (by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => Quiver.Hom.op ((hc (Opposi …
    -/
    simpa using congr_fun ((hc s.pt.unop).fac ((yoneda.obj s.pt.unop).mapCone s) j) (𝟙 _))
    /-
      🎉 no goals
    -/
  uniq s m hm := Quiver.Hom.unop_inj (by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq m.unop ((fun s => Quiver.Hom.op ((hc (Opposite.unop s.pt)).lift ((Categor …
    -/
    apply (Types.isLimitEquivSections (hc s.pt.unop)).injective
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq ((CategoryTheory.Limits.Types.isLimitEquivSections (hc (Opposite.unop s.p …
    -/
    ext j
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      ⊢ Eq (↑((CategoryTheory.Limits.Types.isLimitEquivSections (hc (Opposite.unop s …
    -/
    have eq := congr_fun ((hc s.pt.unop).fac ((yoneda.obj s.pt.unop).mapCone s) j) (𝟙 _)
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp ((hc (Opposite.unop s.pt)).lift (( …
      ⊢ Eq (↑((CategoryTheory.Limits.Types.isLimitEquivSections (hc (Opposite.unop s …
    -/
    dsimp at eq
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).unop ((hc (Opposite.un …
      ⊢ Eq (↑((CategoryTheory.Limits.Types.isLimitEquivSections (hc (Opposite.unop s …
    -/
    dsimp [Types.isLimitEquivSections, Types.sectionOfCone]
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J (Opposite C)
      c : CategoryTheory.Limits.Cone F
      hc : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).unop ((hc (Opposite.un …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j).unop m.unop) (CategoryThe …
    -/
    rw [eq, Category.comp_id, ← hm, unop_comp])
    /-
      🎉 no goals
    -/


/-- A cocone is colimit iff it becomes limit after the
application of `yoneda.obj X` for all `X : C`. -/
noncomputable def Limits.Cocone.isColimitYonedaEquiv {F : J ⥤ C} (c : Cocone F) :
    IsColimit c ≃ ∀ (X : C), IsLimit ((yoneda.obj X).mapCone c.op) where
  toFun h _ := isLimitOfPreserves _ h.op
  invFun h := IsLimit.unop (yonedaJointlyReflectsLimits _ _ h)
  left_inv _ := Subsingleton.elim _ _
                    /-
                      C : Type u
                      inst✝¹ : CategoryTheory.Category.{v, u} C
                      J : Type w
                      inst✝ : CategoryTheory.Category.{t, w} J
                      F : CategoryTheory.Functor J C
                      c : CategoryTheory.Limits.Cocone F
                      x✝ : (X : C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.yoneda.obj X).ma …
                      ⊢ Eq ((fun h x => CategoryTheory.Limits.isLimitOfPreserves (CategoryTheory.yon …
                    -/
  right_inv _ := by ext; apply Subsingleton.elim
                         /-
                           🎉 no goals
                         -/


/-- The cone of `F` corresponding to an element in `(F ⋙ coyoneda.obj X).sections`. -/
@[simps]
def Limits.coneOfSectionCompCoyoneda (F : J ⥤ C) (X : Cᵒᵖ)
    (s : (F ⋙ coyoneda.obj X).sections) : Cone F where
  pt := X.unop
  π := compCoyonedaSectionsEquiv F X.unop s


instance coyoneda_preservesLimit (F : J ⥤ C) (X : Cᵒᵖ) :
    PreservesLimit F (coyoneda.obj X) where
  preserves {c} hc := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      X : Opposite C
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj X).map …
    -/
    rw [Types.isLimit_iff]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      X : Opposite C
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (s : (j : J) → (F.comp (CategoryTheory.coyoneda.obj X)).obj j), Membership …
    -/
    intro s hs
    exact ⟨hc.lift (Limits.coneOfSectionCompCoyoneda F X ⟨s, hs⟩), hc.fac _,
      hc.uniq (Limits.coneOfSectionCompCoyoneda F X ⟨s, hs⟩)⟩


variable (J) in
noncomputable instance coyonedaPreservesLimitsOfShape (X : Cᵒᵖ) :
    PreservesLimitsOfShape J (coyoneda.obj X) where


/-- The coyoneda embeddings jointly reflect limits. -/
def coyonedaJointlyReflectsLimits (F : J ⥤ C) (c : Cone F)
    (hc : ∀ X : Cᵒᵖ, IsLimit ((coyoneda.obj X).mapCone c)) : IsLimit c where
  lift s := (hc (op s.pt)).lift ((coyoneda.obj (op s.pt)).mapCone s) (𝟙 _)
  fac s j := by simpa using congr_fun ((hc (op s.pt)).fac
    ((coyoneda.obj (op s.pt)).mapCone s) j) (𝟙 _)
  uniq s m hm := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq m ((fun s => (hc { unop := s.pt }).lift ((CategoryTheory.coyoneda.obj { u …
    -/
    apply (Types.isLimitEquivSections (hc (op s.pt))).injective
    /-
      case a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      ⊢ Eq ((CategoryTheory.Limits.Types.isLimitEquivSections (hc { unop := s.pt })) …
    -/
    ext j
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      ⊢ Eq (↑((CategoryTheory.Limits.Types.isLimitEquivSections (hc { unop := s.pt } …
    -/
    dsimp [Types.isLimitEquivSections, Types.sectionOfCone]
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (CategoryTheory.Catego …
    -/
    have eq := congr_fun ((hc (op s.pt)).fac ((coyoneda.obj (op s.pt)).mapCone s) j) (𝟙 _)
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp ((hc { unop := s.pt }).lift ((Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (CategoryTheory.Catego …
    -/
    dsimp at eq
    /-
      case a.a.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : Type w
      inst✝ : CategoryTheory.Category.{t, w} J
      F : CategoryTheory.Functor J C
      c : CategoryTheory.Limits.Cone F
      hc : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app …
      j : J
      eq : Eq (CategoryTheory.CategoryStruct.comp ((hc { unop := s.pt }).lift ((Cate …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (CategoryTheory.Catego …
    -/
    rw [eq, Category.id_comp, ← hm]
    /-
      🎉 no goals
    -/


/-- A cone is limit iff it is so after the application of `coyoneda.obj X` for all `X : Cᵒᵖ`. -/
noncomputable def Limits.Cone.isLimitCoyonedaEquiv {F : J ⥤ C} (c : Cone F) :
    IsLimit c ≃ ∀ (X : Cᵒᵖ), IsLimit ((coyoneda.obj X).mapCone c) where
  toFun h _ := isLimitOfPreserves _ h
  invFun h := coyonedaJointlyReflectsLimits _ _ h
  left_inv _ := Subsingleton.elim _ _
                    /-
                      C : Type u
                      inst✝¹ : CategoryTheory.Category.{v, u} C
                      J : Type w
                      inst✝ : CategoryTheory.Category.{t, w} J
                      F : CategoryTheory.Functor J C
                      c : CategoryTheory.Limits.Cone F
                      x✝ : (X : Opposite C) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoned …
                      ⊢ Eq ((fun h x => CategoryTheory.Limits.isLimitOfPreserves (CategoryTheory.coy …
                    -/
  right_inv _ := by ext; apply Subsingleton.elim
                         /-
                           🎉 no goals
                         -/


/-- The yoneda embedding `yoneda.obj X : Cᵒᵖ ⥤ Type v` for `X : C` preserves limits. -/
instance yoneda_preservesLimits (X : C) :
    PreservesLimitsOfSize.{t, w} (yoneda.obj X) where


/-- The coyoneda embedding `coyoneda.obj X : C ⥤ Type v` for `X : Cᵒᵖ` preserves limits. -/
instance coyoneda_preservesLimits (X : Cᵒᵖ) :
    PreservesLimitsOfSize.{t, w} (coyoneda.obj X) where


instance yonedaFunctor_preservesLimits :
    PreservesLimitsOfSize.{t, w} (@yoneda C _) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, max u v, u, max u (v + …
  -/
  apply preservesLimits_of_evaluation
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ (k : Opposite C), CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, …
  -/
  intro K
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    K : Opposite C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, u, v + 1} (Category …
  -/
  change PreservesLimitsOfSize (coyoneda.obj K)
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    K : Opposite C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, u, v + 1} (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance coyonedaFunctor_preservesLimits :
    PreservesLimitsOfSize.{t, w} (@coyoneda C _) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, max u v, u, max u (v + …
  -/
  apply preservesLimits_of_evaluation
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ (k : C), CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, u, v + 1 …
  -/
  intro K
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    K : C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, u, v + 1} (Category …
  -/
  change PreservesLimitsOfSize (yoneda.obj K)
  /-
    case x
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    K : C
    ⊢ CategoryTheory.Limits.PreservesLimitsOfSize.{t, w, v, v, u, v + 1} (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance yonedaFunctor_reflectsLimits :
    ReflectsLimitsOfSize.{t, w} (@yoneda C _) := inferInstance


noncomputable instance coyonedaFunctor_reflectsLimits :
    ReflectsLimitsOfSize.{t, w} (@coyoneda C _) := inferInstance


instance representable_preservesLimit (G : J ⥤ Cᵒᵖ) :
    PreservesLimit G F :=
  preservesLimit_of_natIso _ F.reprW


variable (J) in
instance representable_preservesLimitsOfShape :
    PreservesLimitsOfShape J F where


instance representable_preservesLimits :
    PreservesLimitsOfSize.{t, w} F where


instance corepresentable_preservesLimit (G : J ⥤ C) :
    PreservesLimit G F :=
  preservesLimit_of_natIso _ F.coreprW


variable (J) in
instance corepresentable_preservesLimitsOfShape :
    PreservesLimitsOfShape J F where


instance corepresentable_preservesLimits :
    PreservesLimitsOfSize.{t, w} F where


