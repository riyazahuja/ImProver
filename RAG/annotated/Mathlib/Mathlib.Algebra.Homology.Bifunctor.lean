variable {c₁} in
/-- Auxiliary definition for `mapBifunctorHomologicalComplex`. -/
@[simps!]
def mapBifunctorHomologicalComplexObj (K₁ : HomologicalComplex C₁ c₁) :
    HomologicalComplex C₂ c₂ ⥤ HomologicalComplex₂ D c₁ c₂ where
  obj K₂ := HomologicalComplex₂.ofGradedObject c₁ c₂
      (((GradedObject.mapBifunctor F I₁ I₂).obj K₁.X).obj K₂.X)
      (fun i₁ i₁' i₂ => (F.map (K₁.d i₁ i₁')).app (K₂.X i₂))
      (fun i₁ i₂ i₂' => (F.obj (K₁.X i₁)).map (K₂.d i₂ i₂'))
      (fun i₁ i₁' h₁ i₂ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ i₁' : I₁
          h₁ : Not (c₁.Rel i₁ i₁')
          i₂ : I₂
          ⊢ Eq ((fun i₁ i₁' i₂ => (F.map (K₁.d i₁ i₁')).app (K₂.X i₂)) i₁ i₁' i₂) 0
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ i₁' : I₁
          h₁ : Not (c₁.Rel i₁ i₁')
          i₂ : I₂
          ⊢ Eq ((F.map (K₁.d i₁ i₁')).app (K₂.X i₂)) 0
        -/
        rw [K₁.shape _ _ h₁, Functor.map_zero, zero_app])
        /-
          🎉 no goals
        -/
      (fun i₁ i₂ i₂' h₂ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ : I₁
          i₂ i₂' : I₂
          h₂ : Not (c₂.Rel i₂ i₂')
          ⊢ Eq ((fun i₁ i₂ i₂' => (F.obj (K₁.X i₁)).map (K₂.d i₂ i₂')) i₁ i₂ i₂') 0
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ : I₁
          i₂ i₂' : I₂
          h₂ : Not (c₂.Rel i₂ i₂')
          ⊢ Eq ((F.obj (K₁.X i₁)).map (K₂.d i₂ i₂')) 0
        -/
        rw [K₂.shape _ _ h₂, Functor.map_zero])
        /-
          🎉 no goals
        -/
      (fun i₁ i₁' i₁'' i₂ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ i₁' i₁'' : I₁
          i₂ : I₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ i₁' i₂ => (F.map (K₁.d i₁ i₁ …
        -/
        dsimp
        rw [← NatTrans.comp_app, ← Functor.map_comp, HomologicalComplex.d_comp_d,
          Functor.map_zero, zero_app])
      (fun i₁ i₂ i₂' i₂'' => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ : I₁
          i₂ i₂' i₂'' : I₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ i₂ i₂' => (F.obj (K₁.X i₁)). …
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ : I₁
          i₂ i₂' i₂'' : I₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.d i₂ i₂')) …
        -/
        rw [← Functor.map_comp, HomologicalComplex.d_comp_d, Functor.map_zero])
        /-
          🎉 no goals
        -/
      (fun i₁ i₁' i₂ i₂' => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ i₁' : I₁
          i₂ i₂' : I₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ i₁' i₂ => (F.map (K₁.d i₁ i₁ …
        -/
        dsimp
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D : Type u_3
          inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
          I₁ : Type u_4
          I₂ : Type u_5
          J : Type u_6
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          inst✝¹ : F.PreservesZeroMorphisms
          inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          i₁ i₁' : I₁
          i₂ i₂' : I₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (K₁.d i₁ i₁')).app (K₂.X i₂)) …
        -/
        rw [NatTrans.naturality])
        /-
          🎉 no goals
        -/
  map {K₂ K₂' φ} := HomologicalComplex₂.homMk
      (((GradedObject.mapBifunctor F I₁ I₂).obj K₁.X).map φ.f)
            /-
              C₁ : Type u_1
              C₂ : Type u_2
              D : Type u_3
              inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
              inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
              inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
              inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
              I₁ : Type u_4
              I₂ : Type u_5
              J : Type u_6
              c₁ : ComplexShape I₁
              c₂ : ComplexShape I₂
              inst✝¹ : F.PreservesZeroMorphisms
              inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
              K₁ : HomologicalComplex C₁ c₁
              K₂ K₂' : HomologicalComplex C₂ c₂
              φ : Quiver.Hom K₂ K₂'
              ⊢ ∀ (i₁ i₁' : I₁) (i₂ : I₂), c₁.Rel i₁ i₁' → Eq (CategoryTheory.CategoryStruct …
            -/
        (by dsimp; intros; rw [NatTrans.naturality]) (by
                           /-
                             🎉 no goals
                           -/
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ : HomologicalComplex C₁ c₁
            K₂ K₂' : HomologicalComplex C₂ c₂
            φ : Quiver.Hom K₂ K₂'
            ⊢ ∀ (i₁ : I₁) (i₂ i₂' : I₂), c₂.Rel i₂ i₂' → Eq (CategoryTheory.CategoryStruct …
          -/
          dsimp
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ : HomologicalComplex C₁ c₁
            K₂ K₂' : HomologicalComplex C₂ c₂
            φ : Quiver.Hom K₂ K₂'
            ⊢ ∀ (i₁ : I₁) (i₂ i₂' : I₂), c₂.Rel i₂ i₂' → Eq (CategoryTheory.CategoryStruct …
          -/
          intros
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ : HomologicalComplex C₁ c₁
            K₂ K₂' : HomologicalComplex C₂ c₂
            φ : Quiver.Hom K₂ K₂'
            i₁✝ : I₁
            i₂✝ i₂'✝ : I₂
            a✝ : c₂.Rel i₂✝ i₂'✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁✝)).map (φ.f i₂✝)) (( …
          -/
          simp only [← Functor.map_comp, φ.comm])
          /-
            🎉 no goals
          -/
                  /-
                    C₁ : Type u_1
                    C₂ : Type u_2
                    D : Type u_3
                    inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
                    inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
                    inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
                    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
                    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
                    inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
                    I₁ : Type u_4
                    I₂ : Type u_5
                    J : Type u_6
                    c₁ : ComplexShape I₁
                    c₂ : ComplexShape I₂
                    inst✝¹ : F.PreservesZeroMorphisms
                    inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
                    K₁ : HomologicalComplex C₁ c₁
                    K₂ : HomologicalComplex C₂ c₂
                    ⊢ Eq ({ obj := fun K₂ => HomologicalComplex₂.ofGradedObject c₁ c₂ (((CategoryT …
                  -/
  map_id K₂ := by dsimp; ext; dsimp; rw [Functor.map_id]
                                     /-
                                       🎉 no goals
                                     -/
                     /-
                       C₁ : Type u_1
                       C₂ : Type u_2
                       D : Type u_3
                       inst✝⁷ : CategoryTheory.Category.{?u.883, u_1} C₁
                       inst✝⁶ : CategoryTheory.Category.{?u.887, u_2} C₂
                       inst✝⁵ : CategoryTheory.Category.{?u.891, u_3} D
                       inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                       F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
                       I₁ : Type u_4
                       I₂ : Type u_5
                       J : Type u_6
                       c₁ : ComplexShape I₁
                       c₂ : ComplexShape I₂
                       inst✝¹ : F.PreservesZeroMorphisms
                       inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
                       K₁ : HomologicalComplex C₁ c₁
                       X✝ Y✝ Z✝ : HomologicalComplex C₂ c₂
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun K₂ => HomologicalComplex₂.ofGradedObject c₁ c₂ (((CategoryT …
                     -/
  map_comp f g := by dsimp; ext; dsimp; rw [Functor.map_comp]
                                        /-
                                          🎉 no goals
                                        -/


/-- Given a functor `F : C₁ ⥤ C₂ ⥤ D`, this is the bifunctor which sends
`K₁ : HomologicalComplex C₁ c₁` and `K₂ : HomologicalComplex C₂ c₂` to the bicomplex
which is degree `(i₁, i₂)` consists of `(F.obj (K₁.X i₁)).obj (K₂.X i₂)`. -/
@[simps! obj_obj_X_X obj_obj_X_d obj_obj_d_f obj_map_f_f map_app_f_f]
def mapBifunctorHomologicalComplex :
    HomologicalComplex C₁ c₁ ⥤ HomologicalComplex C₂ c₂ ⥤ HomologicalComplex₂ D c₁ c₂ where
  obj := mapBifunctorHomologicalComplexObj F c₂
  map {K₁ K₁'} f :=
    { app := fun K₂ => HomologicalComplex₂.homMk
        (((GradedObject.mapBifunctor F I₁ I₂).map f.f).app K₂.X) (by
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.35463, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.35467, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.35471, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ K₁' : HomologicalComplex C₁ c₁
            f : Quiver.Hom K₁ K₁'
            K₂ : HomologicalComplex C₂ c₂
            ⊢ ∀ (i₁ i₁' : I₁) (i₂ : I₂), c₁.Rel i₁ i₁' → Eq (CategoryTheory.CategoryStruct …
          -/
          intros
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.35463, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.35467, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.35471, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ K₁' : HomologicalComplex C₁ c₁
            f : Quiver.Hom K₁ K₁'
            K₂ : HomologicalComplex C₂ c₂
            i₁✝ i₁'✝ : I₁
            i₂✝ : I₂
            a✝ : c₁.Rel i₁✝ i₁'✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.GradedObject.mapBif …
          -/
          dsimp
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            D : Type u_3
            inst✝⁷ : CategoryTheory.Category.{?u.35463, u_1} C₁
            inst✝⁶ : CategoryTheory.Category.{?u.35467, u_2} C₂
            inst✝⁵ : CategoryTheory.Category.{?u.35471, u_3} D
            inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
            I₁ : Type u_4
            I₂ : Type u_5
            J : Type u_6
            c₁ : ComplexShape I₁
            c₂ : ComplexShape I₂
            inst✝¹ : F.PreservesZeroMorphisms
            inst✝ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
            K₁ K₁' : HomologicalComplex C₁ c₁
            f : Quiver.Hom K₁ K₁'
            K₂ : HomologicalComplex C₂ c₂
            i₁✝ i₁'✝ : I₁
            i₂✝ : I₂
            a✝ : c₁.Rel i₁✝ i₁'✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map (f.f i₁✝)).app (K₂.X i₂✝)) (( …
          -/
          /-
            🎉 no goals
          -/
          simp only [← NatTrans.comp_app, ← F.map_comp, f.comm]) (by simp) }
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
lemma mapBifunctorHomologicalComplex_obj_obj_toGradedObject
    (K₁ : HomologicalComplex C₁ c₁) (K₂ : HomologicalComplex C₂ c₂) :
    (((mapBifunctorHomologicalComplex F c₁ c₂).obj K₁).obj K₂).toGradedObject =
      ((GradedObject.mapBifunctor F I₁ I₂).obj K₁.X).obj K₂.X := rfl


/-- The condition that `((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂` has
a total complex. -/
abbrev HasMapBifunctor := (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).HasTotal c


/-- Given `K₁ : HomologicalComplex C₁ c₁`, `K₂ : HomologicalComplex C₂ c₂`,
a bifunctor `F : C₁ ⥤ C₂ ⥤ D` and a complex shape `ComplexShape J` such that we have
`[TotalComplexShape c₁ c₂ c]`, this `mapBifunctor K₁ K₂ F c : HomologicalComplex D c`
is the total complex of the bicomplex obtained by applying `F` to `K₁` and `K₂`. -/
noncomputable abbrev mapBifunctor : HomologicalComplex D c :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).total c


/-- The inclusion of a summand of `(mapBifunctor K₁ K₂ F c).X j`. -/
noncomputable abbrev ιMapBifunctor
    (i₁ : I₁) (i₂ : I₂) (j : J) (h : ComplexShape.π c₁ c₂ c (i₁, i₂) = j) :
    (F.obj (K₁.X i₁)).obj (K₂.X i₂) ⟶ (mapBifunctor K₁ K₂ F c).X j :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).ιTotal c i₁ i₂ j h


/-- The inclusion of a summand of `(mapBifunctor K₁ K₂ F c).X j`, or zero. -/
noncomputable abbrev ιMapBifunctorOrZero (i₁ : I₁) (i₂ : I₂) (j : J) :
    (F.obj (K₁.X i₁)).obj (K₂.X i₂) ⟶ (mapBifunctor K₁ K₂ F c).X j :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).ιTotalOrZero c i₁ i₂ j


lemma ιMapBifunctorOrZero_eq (i₁ : I₁) (i₂ : I₂) (j : J)
    (h : ComplexShape.π c₁ c₂ c (i₁, i₂) = j) :
    ιMapBifunctorOrZero K₁ K₂ F c i₁ i₂ j = ιMapBifunctor K₁ K₂ F c i₁ i₂ j h := dif_pos h


lemma ιMapBifunctorOrZero_eq_zero (i₁ : I₁) (i₂ : I₂) (j : J)
    (h : ComplexShape.π c₁ c₂ c (i₁, i₂) ≠ j) :
    ιMapBifunctorOrZero K₁ K₂ F c i₁ i₂ j = 0 := dif_neg h


/-- Constructor for morphisms from `(mapBifunctor K₁ K₂ F c).X j`. -/
noncomputable def mapBifunctorDesc : (mapBifunctor K₁ K₂ F c).X j ⟶ A :=
  HomologicalComplex₂.totalDesc _ f


@[reassoc (attr := simp)]
lemma ι_mapBifunctorDesc (i₁ : I₁) (i₂ : I₂) (h : ComplexShape.π c₁ c₂ c ⟨i₁, i₂⟩ = j) :
    ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ mapBifunctorDesc f = f i₁ i₂ h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_7, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁵ : CategoryTheory.Preadditive D
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁴ : F.PreservesZeroMorphisms
    inst✝³ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : DecidableEq J
    A : D
    j : J
    f : (i₁ : I₁) → (i₂ : I₂) → Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j → Quiver …
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h) ( …
  -/
  apply HomologicalComplex₂.ι_totalDesc
  /-
    🎉 no goals
  -/


variable {K₁ K₂ F c} in
@[ext]
lemma hom_ext {Y : D} {j : J} {f g : (mapBifunctor K₁ K₂ F c).X j ⟶ Y}
    (h : ∀ (i₁ : I₁) (i₂ : I₂) (h : ComplexShape.π c₁ c₂ c ⟨i₁, i₂⟩ = j),
      ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ f = ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ g) :
    f = g :=
  HomologicalComplex₂.total.hom_ext _ h


/-- The first differential on `mapBifunctor K₁ K₂ F c` -/
noncomputable def D₁ :
    (mapBifunctor K₁ K₂ F c).X j ⟶ (mapBifunctor K₁ K₂ F c).X j' :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).D₁ c j j'


/-- The second differential on `mapBifunctor K₁ K₂ F c` -/
noncomputable def D₂ :
    (mapBifunctor K₁ K₂ F c).X j ⟶ (mapBifunctor K₁ K₂ F c).X j' :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).D₂ c j j'


lemma d_eq :
    (mapBifunctor K₁ K₂ F c).d j j' = D₁ K₁ K₂ F c j j' + D₂ K₁ K₂ F c j j' := rfl


/-- The first differential on a summand of `mapBifunctor K₁ K₂ F c` -/
noncomputable def d₁ :
    (F.obj (K₁.X i₁)).obj (K₂.X i₂) ⟶ (mapBifunctor K₁ K₂ F c).X j :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).d₁ c i₁ i₂ j


/-- The second differential on a summand of `mapBifunctor K₁ K₂ F c` -/
noncomputable def d₂ :
    (F.obj (K₁.X i₁)).obj (K₂.X i₂) ⟶ (mapBifunctor K₁ K₂ F c).X j :=
  (((F.mapBifunctorHomologicalComplex c₁ c₂).obj K₁).obj K₂).d₂ c i₁ i₂ j


lemma d₁_eq_zero (h : ¬ c₁.Rel i₁ (c₁.next i₁)):
    d₁ K₁ K₂ F c i₁ i₂ j = 0 :=
  HomologicalComplex₂.d₁_eq_zero _ _ _ _ _ h


lemma d₂_eq_zero (h : ¬ c₂.Rel i₂ (c₂.next i₂)):
    d₂ K₁ K₂ F c i₁ i₂ j = 0 :=
  HomologicalComplex₂.d₂_eq_zero _ _ _ _ _ h


lemma d₁_eq_zero' {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (j : J)
    (h' : ComplexShape.π c₁ c₂ c ⟨i₁', i₂⟩ ≠ j) :
    d₁ K₁ K₂ F c i₁ i₂ j = 0 :=
  HomologicalComplex₂.d₁_eq_zero' _ _ h _ _ h'


lemma d₂_eq_zero' (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (j : J)
    (h' : ComplexShape.π c₁ c₂ c ⟨i₁, i₂'⟩ ≠ j) :
    d₂ K₁ K₂ F c i₁ i₂ j = 0 :=
  HomologicalComplex₂.d₂_eq_zero' _ _ _ h _ h'


lemma d₁_eq' {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (j : J) :
    d₁ K₁ K₂ F c i₁ i₂ j = ComplexShape.ε₁ c₁ c₂ c ⟨i₁, i₂⟩ •
      ((F.map (K₁.d i₁ i₁')).app (K₂.X i₂) ≫ ιMapBifunctorOrZero K₁ K₂ F c i₁' i₂ j) :=
  HomologicalComplex₂.d₁_eq' _ _ h _ _


lemma d₂_eq' (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (j : J) :
    d₂ K₁ K₂ F c i₁ i₂ j = ComplexShape.ε₂ c₁ c₂ c ⟨i₁, i₂⟩ •
      ((F.obj (K₁.X i₁)).map (K₂.d i₂ i₂') ≫ ιMapBifunctorOrZero K₁ K₂ F c i₁ i₂' j) :=
  HomologicalComplex₂.d₂_eq' _ _ _ h _


lemma d₁_eq {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) (j : J)
    (h' : ComplexShape.π c₁ c₂ c ⟨i₁', i₂⟩ = j) :
    d₁ K₁ K₂ F c i₁ i₂ j = ComplexShape.ε₁ c₁ c₂ c ⟨i₁, i₂⟩ •
      ((F.map (K₁.d i₁ i₁')).app (K₂.X i₂) ≫ ιMapBifunctor K₁ K₂ F c i₁' i₂ j h') :=
  HomologicalComplex₂.d₁_eq _ _ h _ _ h'


lemma d₂_eq (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') (j : J)
    (h' : ComplexShape.π c₁ c₂ c ⟨i₁, i₂'⟩ = j) :
    d₂ K₁ K₂ F c i₁ i₂ j = ComplexShape.ε₂ c₁ c₂ c ⟨i₁, i₂⟩ •
      ((F.obj (K₁.X i₁)).map (K₂.d i₂ i₂') ≫ ιMapBifunctor K₁ K₂ F c i₁ i₂' j h') :=
  HomologicalComplex₂.d₂_eq _ _ _ h _ h'


@[reassoc (attr := simp)]
lemma ι_D₁ :
    ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ D₁ K₁ K₂ F c j j' = d₁ K₁ K₂ F c i₁ i₂ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_7, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁵ : CategoryTheory.Preadditive D
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁴ : F.PreservesZeroMorphisms
    inst✝³ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : DecidableEq J
    j j' : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h) ( …
  -/
  apply HomologicalComplex₂.ι_D₁
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_D₂ :
    ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ D₂ K₁ K₂ F c j j' = d₂ K₁ K₂ F c i₁ i₂ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹⁰ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝⁹ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁸ : CategoryTheory.Category.{u_7, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁵ : CategoryTheory.Preadditive D
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁴ : F.PreservesZeroMorphisms
    inst✝³ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝² : TotalComplexShape c₁ c₂ c
    inst✝¹ : K₁.HasMapBifunctor K₂ F c
    inst✝ : DecidableEq J
    j j' : J
    i₁ : I₁
    i₂ : I₂
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h) ( …
  -/
  apply HomologicalComplex₂.ι_D₂
  /-
    🎉 no goals
  -/


/-- The morphism `mapBifunctor K₁ K₂ F c ⟶ mapBifunctor L₁ L₂ F c` induced by
morphisms of complexes `K₁ ⟶ L₁` and `K₂ ⟶ L₂`. -/
noncomputable def mapBifunctorMap : mapBifunctor K₁ K₂ F c ⟶ mapBifunctor L₁ L₂ F c :=
  HomologicalComplex₂.total.map (((F.mapBifunctorHomologicalComplex c₁ c₂).map f₁).app K₂ ≫
    ((F.mapBifunctorHomologicalComplex c₁ c₂).obj L₁).map f₂) c


@[reassoc (attr := simp)]
lemma ι_mapBifunctorMap (i₁ : I₁) (i₂ : I₂) (j : J)
    (h : ComplexShape.π c₁ c₂ c (i₁, i₂) = j) :
    ιMapBifunctor K₁ K₂ F c i₁ i₂ j h ≫ (mapBifunctorMap f₁ f₂ F c).f j =
      (F.map (f₁.f i₁)).app (K₂.X i₂) ≫ (F.obj (L₁.X i₁)).map (f₂.f i₂) ≫
        ιMapBifunctor L₁ L₂ F c i₁ i₂ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D : Type u_3
    inst✝¹¹ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝¹⁰ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁹ : CategoryTheory.Category.{u_7, u_3} D
    I₁ : Type u_4
    I₂ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    inst✝⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝⁶ : CategoryTheory.Preadditive D
    K₁ L₁ : HomologicalComplex C₁ c₁
    K₂ L₂ : HomologicalComplex C₂ c₂
    f₁ : Quiver.Hom K₁ L₁
    f₂ : Quiver.Hom K₂ L₂
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ D)
    inst✝⁵ : F.PreservesZeroMorphisms
    inst✝⁴ : ∀ (X₁ : C₁), (F.obj X₁).PreservesZeroMorphisms
    c : ComplexShape J
    inst✝³ : TotalComplexShape c₁ c₂ c
    inst✝² : K₁.HasMapBifunctor K₂ F c
    inst✝¹ : L₁.HasMapBifunctor L₂ F c
    inst✝ : DecidableEq J
    i₁ : I₁
    i₂ : I₂
    j : J
    h : Eq (c₁.π c₂ c { fst := i₁, snd := i₂ }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K₁.ιMapBifunctor K₂ F c i₁ i₂ j h) ( …
  -/
  simp [mapBifunctorMap]
  /-
    🎉 no goals
  -/


