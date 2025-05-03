/-- If `C` has finite coproducts, a functor `Discrete α ⥤ C` lifts to a functor
    `Finset (Discrete α) ⥤ C` by taking coproducts. -/
@[simps!]
def liftToFinsetObj (F : Discrete α ⥤ C) : Finset (Discrete α) ⥤ C where
  obj s := ∐ fun x : s => F.obj x
  map {_ Y} h := Sigma.desc fun y =>
    Sigma.ι (fun (x : { x // x ∈ Y }) => F.obj x) ⟨y, h.down.down y.2⟩


/-- If `C` has finite coproducts and filtered colimits, we can construct arbitrary coproducts by
    taking the colimit of the diagram formed by the coproducts of finite sets over the indexing
    type. -/
@[simps!]
def liftToFinsetColimitCocone [HasColimitsOfShape (Finset (Discrete α)) C]
    (F : Discrete α ⥤ C) : ColimitCocone F where
  cocone :=
    { pt := colimit (liftToFinsetObj F)
      ι :=
        Discrete.natTrans fun j =>
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                                                     α : Type w
                                                                                     inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                                     inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
                                                                                     F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
                                                                                     j : CategoryTheory.Discrete α
                                                                                     ⊢ Membership.mem (Singleton.singleton j) j
                                                                                   -/
          @Sigma.ι _ _ _ (fun x : ({j} : Finset (Discrete α)) => F.obj x) _ ⟨j, by simp⟩ ≫
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
            colimit.ι (liftToFinsetObj F) {j} }
  isColimit :=
    { desc := fun s =>
        colimit.desc (liftToFinsetObj F)
          { pt := s.pt
            ι := { app := fun _ => Sigma.desc fun x => s.ι.app x } }
      uniq := fun s m h => by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m ((fun s => CategoryTheory.Limits.colimit.desc (CategoryTheory.Limits.Co …
        -/
        apply colimit.hom_ext
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ ∀ (j : Finset (CategoryTheory.Discrete α)), Eq (CategoryTheory.CategoryStruc …
        -/
        rintro t
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Finset (CategoryTheory.Discrete α)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
        -/
        dsimp [liftToFinsetObj]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Finset (CategoryTheory.Discrete α)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι { ob …
        -/
        apply colimit.hom_ext
        /-
          case w.w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Finset (CategoryTheory.Discrete α)
          ⊢ ∀ (j : CategoryTheory.Discrete (Subtype fun x => Membership.mem t x)), Eq (C …
        -/
        rintro ⟨⟨j, hj⟩⟩
        /-
          case w.w.mk.mk
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cocone F
          m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Finset (CategoryTheory.Discrete α)
          j : CategoryTheory.Discrete α
          hj : Membership.mem t j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
        -/
        convert h j using 1
          /-
            case h.e'_2.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cocone F
            m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Finset (CategoryTheory.Discrete α)
            j : CategoryTheory.Discrete α
            hj : Membership.mem t j
            e_1✝ : Eq (Quiver.Hom ((CategoryTheory.Discrete.functor fun x => F.obj ↑x).obj …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
          -/
        · simp [← colimit.w (liftToFinsetObj F) ⟨⟨Finset.singleton_subset_iff.2 hj⟩⟩]
          /-
            case h.e'_2.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cocone F
            m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Finset (CategoryTheory.Discrete α)
            j : CategoryTheory.Discrete α
            hj : Membership.mem t j
            e_1✝ : Eq (Quiver.Hom ((CategoryTheory.Discrete.functor fun x => F.obj ↑x).obj …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Discr …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cocone F
            m : Quiver.Hom { pt := CategoryTheory.Limits.colimit (CategoryTheory.Limits.Co …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Finset (CategoryTheory.Discrete α)
            j : CategoryTheory.Discrete α
            hj : Membership.mem t j
            e_1✝ : Eq (Quiver.Hom ((CategoryTheory.Discrete.functor fun x => F.obj ↑x).obj …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
          -/
        · aesop_cat }
          /-
            🎉 no goals
          -/


/-- The functor taking a functor `Discrete α ⥤ C` to a functor `Finset (Discrete α) ⥤ C` by taking
coproducts. -/
@[simps!]
def liftToFinset : (Discrete α ⥤ C) ⥤ (Finset (Discrete α) ⥤ C) where
  obj := liftToFinsetObj
  map := fun β => { app := fun _ => Sigma.map (fun x => β.app x.val) }


theorem hasCoproducts_of_finite_and_filtered [HasFiniteCoproducts C]
    [HasFilteredColimitsOfSize.{w, w} C] : HasCoproducts.{w} C := fun α => by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v, u} C
    α : Type w
    ⊢ CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
  -/
  classical exact ⟨fun F => HasColimit.mk (liftToFinsetColimitCocone F)⟩
  /-
    🎉 no goals
  -/


theorem has_colimits_of_finite_and_filtered [HasFiniteColimits C]
    [HasFilteredColimitsOfSize.{w, w} C] : HasColimitsOfSize.{w, w} C :=
  have : HasCoproducts.{w} C := hasCoproducts_of_finite_and_filtered
  has_colimits_of_hasCoequalizers_and_coproducts


theorem hasProducts_of_finite_and_cofiltered [HasFiniteProducts C]
    [HasCofilteredLimitsOfSize.{w, w} C] : HasProducts.{w} C :=
  have : HasCoproducts.{w} Cᵒᵖ := hasCoproducts_of_finite_and_filtered
  hasProducts_of_opposite


theorem has_limits_of_finite_and_cofiltered [HasFiniteLimits C]
    [HasCofilteredLimitsOfSize.{w, w} C] : HasLimitsOfSize.{w, w} C :=
  have : HasProducts.{w} C := hasProducts_of_finite_and_cofiltered
  has_limits_of_hasEqualizers_and_products


/-- Helper construction for `liftToFinsetColimIso`. -/
@[reassoc]
theorem liftToFinsetColimIso_aux (F : Discrete α ⥤ C) {J : Finset (Discrete α)} (j : J) :
    Sigma.ι (F.obj ·.val) j ≫ colimit.ι (liftToFinsetObj F) J ≫
      (colimit.isoColimitCocone (liftToFinsetColimitCocone F)).inv
    = colimit.ι F j := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    α : Type w
    inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Disc …
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
    F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
    J : Finset (CategoryTheory.Discrete α)
    j : Subtype fun x => Membership.mem J x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun x …
  -/
  simp [colimit.isoColimitCocone, IsColimit.coconePointUniqueUpToIso]
  /-
    🎉 no goals
  -/


/-- The `liftToFinset` functor, precomposed with forming a colimit, is a coproduct on the original
functor. -/
def liftToFinsetColimIso : liftToFinset C α ⋙ colim ≅ colim :=
  NatIso.ofComponents
    (fun F => Iso.symm <| colimit.isoColimitCocone (liftToFinsetColimitCocone F))
    (fun β => by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Disc …
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.CoproductsFr …
      -/
      simp only [Functor.comp_obj, colim_obj, Functor.comp_map, colim_map, Iso.symm_hom]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Disc …
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimMap ((Cat …
      -/
      ext J
      /-
        case w
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Disc …
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        J : Finset (CategoryTheory.Discrete α)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
      -/
      simp only [liftToFinset_obj_obj, liftToFinset_map_app]
      /-
        case w
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
        inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Finset (CategoryTheory.Disc …
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        J : Finset (CategoryTheory.Discrete α)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
      -/
      ext j
      simp only [liftToFinset, ι_colimMap_assoc, liftToFinsetObj_obj, Discrete.functor_obj_eq_as,
        Discrete.natTrans_app, liftToFinsetColimIso_aux, liftToFinsetColimIso_aux_assoc,
        ι_colimMap])


/-- `liftToFinset`, when composed with the evaluation functor, results in the whiskering composed
with `colim`. -/
def liftToFinsetEvaluationIso [HasFiniteCoproducts C] (I : Finset (Discrete α)) :
    liftToFinset C α ⋙ (evaluation _ _).obj I ≅
    (whiskeringLeft _ _ _).obj (Discrete.functor (·.val)) ⋙ colim (J := Discrete I) :=
  NatIso.ofComponents (fun _ => HasColimit.isoOfNatIso (Discrete.natIso fun _ => Iso.refl _))
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  α : Type w
                  inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
                  I : Finset (CategoryTheory.Discrete α)
                  X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
                  x✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.CoproductsFr …
                -/
    fun _ => by dsimp; ext; simp
                            /-
                              🎉 no goals
                            -/


/-- If `C` has finite coproducts, a functor `Discrete α ⥤ C` lifts to a functor
    `Finset (Discrete α) ⥤ C` by taking coproducts. -/
@[simps!]
def liftToFinsetObj (F : Discrete α ⥤ C) : (Finset (Discrete α))ᵒᵖ ⥤ C where
  obj s := ∏ᶜ (fun x : s.unop => F.obj x)
  map {Y _} h := Pi.lift fun y =>
    Pi.π (fun (x : { x // x ∈ Y.unop }) => F.obj x) ⟨y, h.unop.down.down y.2⟩



/-- If `C` has finite coproducts and filtered colimits, we can construct arbitrary coproducts by
    taking the colimit of the diagram formed by the coproducts of finite sets over the indexing
    type. -/
@[simps!]
def liftToFinsetLimitCone [HasLimitsOfShape (Finset (Discrete α))ᵒᵖ C]
    (F : Discrete α ⥤ C) : LimitCone F where
  cone :=
    { pt := limit (liftToFinsetObj F)
      π := Discrete.natTrans fun j =>
                                                           /-
                                                             C : Type u
                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                             α : Type w
                                                             inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
                                                             inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
                                                             F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
                                                             j : CategoryTheory.Discrete α
                                                             ⊢ Membership.mem (Singleton.singleton j) j
                                                           -/
        limit.π (liftToFinsetObj F) ⟨{j}⟩ ≫ Pi.π _ (⟨j, by simp⟩ : ({j} : Finset (Discrete α))) }
                                                           /-
                                                             🎉 no goals
                                                           -/
  isLimit :=
    { lift := fun s =>
        limit.lift (liftToFinsetObj F)
          { pt := s.pt
            π := { app := fun _ => Pi.lift fun x => s.π.app x } }
      uniq := fun s m h => by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m ((fun s => CategoryTheory.Limits.limit.lift (CategoryTheory.Limits.Prod …
        -/
        apply limit.hom_ext
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ ∀ (j : Opposite (Finset (CategoryTheory.Discrete α))), Eq (CategoryTheory.Ca …
        -/
        rintro t
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Opposite (Finset (CategoryTheory.Discrete α))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.limit.π (Cat …
        -/
        dsimp [liftToFinsetObj]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Opposite (Finset (CategoryTheory.Discrete α))
          ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.limit.π { ob …
        -/
        apply limit.hom_ext
        /-
          case w.w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Opposite (Finset (CategoryTheory.Discrete α))
          ⊢ ∀ (j : CategoryTheory.Discrete (Subtype fun x => Membership.mem (Opposite.un …
        -/
        rintro ⟨⟨j, hj⟩⟩
        /-
          case w.w.mk.mk
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          α : Type w
          inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
          inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
          F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
          s : CategoryTheory.Limits.Cone F
          m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
          h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
          t : Opposite (Finset (CategoryTheory.Discrete α))
          j : CategoryTheory.Discrete α
          hj : Membership.mem (Opposite.unop t) j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
        -/
        convert h j using 1
          /-
            case h.e'_2.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cone F
            m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Opposite (Finset (CategoryTheory.Discrete α))
            j : CategoryTheory.Discrete α
            hj : Membership.mem (Opposite.unop t) j
            e_1✝ : Eq (Quiver.Hom s.pt ((CategoryTheory.Discrete.functor fun x => F.obj ↑x …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp m …
          -/
        · simp [← limit.w (liftToFinsetObj F) ⟨⟨⟨Finset.singleton_subset_iff.2 hj⟩⟩⟩]
          /-
            case h.e'_2.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cone F
            m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Opposite (Finset (CategoryTheory.Discrete α))
            j : CategoryTheory.Discrete α
            hj : Membership.mem (Opposite.unop t) j
            e_1✝ : Eq (Quiver.Hom s.pt ((CategoryTheory.Discrete.functor fun x => F.obj ↑x …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3.h
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            α : Type w
            inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
            inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryTheo …
            F : CategoryTheory.Functor (CategoryTheory.Discrete α) C
            s : CategoryTheory.Limits.Cone F
            m : Quiver.Hom s.pt { pt := CategoryTheory.Limits.limit (CategoryTheory.Limits …
            h : ∀ (j : CategoryTheory.Discrete α), Eq (CategoryTheory.CategoryStruct.comp  …
            t : Opposite (Finset (CategoryTheory.Discrete α))
            j : CategoryTheory.Discrete α
            hj : Membership.mem (Opposite.unop t) j
            e_1✝ : Eq (Quiver.Hom s.pt ((CategoryTheory.Discrete.functor fun x => F.obj ↑x …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · aesop_cat }
          /-
            🎉 no goals
          -/


/-- The functor taking a functor `Discrete α ⥤ C` to a functor `Finset (Discrete α) ⥤ C` by taking
coproducts. -/
@[simps!]
def liftToFinset : (Discrete α ⥤ C) ⥤ ((Finset (Discrete α))ᵒᵖ ⥤ C) where
  obj := liftToFinsetObj
  map := fun β => { app := fun _ => Pi.map (fun x => β.app x.val) }


/-- The `liftToFinset` functor, precomposed with forming a colimit, is a coproduct on the original
functor. -/
def liftToFinsetLimIso [HasLimitsOfShape (Finset (Discrete α))ᵒᵖ C]
    [HasLimitsOfShape (Discrete α) C] : liftToFinset C α ⋙ lim ≅ lim :=
  NatIso.ofComponents
    (fun F => Iso.symm <| limit.isoLimitCone (liftToFinsetLimitCone F))
    (fun β => by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryThe …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.ProductsFrom …
      -/
      simp only [Functor.comp_obj, lim_obj, Functor.comp_map, lim_map, Iso.symm_hom]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryThe …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limMap ((Categ …
      -/
      ext J
      /-
        case w
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        α : Type w
        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
        inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Opposite (Finset (CategoryThe …
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
        X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
        β : Quiver.Hom X✝ Y✝
        J : CategoryTheory.Discrete α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp [liftToFinset])
      /-
        🎉 no goals
      -/


/-- `liftToFinset`, when composed with the evaluation functor, results in the whiskering composed
with `colim`. -/
def liftToFinsetEvaluationIso (I : Finset (Discrete α)) :
    liftToFinset C α ⋙ (evaluation _ _).obj ⟨I⟩ ≅
    (whiskeringLeft _ _ _).obj (Discrete.functor (·.val)) ⋙ lim (J := Discrete I) :=
  NatIso.ofComponents (fun _ => HasLimit.isoOfNatIso (Discrete.natIso fun _ => Iso.refl _))
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  α : Type w
                  inst✝ : CategoryTheory.Limits.HasFiniteProducts C
                  I : Finset (CategoryTheory.Discrete α)
                  X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete α) C
                  x✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.ProductsFrom …
                -/
    fun _ => by dsimp; ext; simp
                            /-
                              🎉 no goals
                            -/


