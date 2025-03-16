/-- Define what it means for a functor `F : C ⥤ D` to reflect isomorphisms: for any
morphism `f : A ⟶ B`, if `F.map f` is an isomorphism then `f` is as well.
Note that we do not assume or require that `F` is faithful.
-/
class Functor.ReflectsIsomorphisms (F : C ⥤ D) : Prop where
  /-- For any `f`, if `F.map f` is an iso, then so was `f`-/
  reflects : ∀ {A B : C} (f : A ⟶ B) [IsIso (F.map f)], IsIso f


@[deprecated (since := "2024-04-06")] alias ReflectsIsomorphisms := Functor.ReflectsIsomorphisms


/-- If `F` reflects isos and `F.map f` is an iso, then `f` is an iso. -/
theorem isIso_of_reflects_iso {A B : C} (f : A ⟶ B) (F : C ⥤ D) [IsIso (F.map f)]
    [F.ReflectsIsomorphisms] : IsIso f :=
  ReflectsIsomorphisms.reflects F f


lemma isIso_iff_of_reflects_iso {A B : C} (f : A ⟶ B) (F : C ⥤ D) [F.ReflectsIsomorphisms] :
    IsIso (F.map f) ↔ IsIso f :=
  ⟨fun _ => isIso_of_reflects_iso f F, fun _ => inferInstance⟩


lemma Functor.FullyFaithful.reflectsIsomorphisms {F : C ⥤ D} (hF : F.FullyFaithful) :
    F.ReflectsIsomorphisms where
  reflects _ _ := hF.isIso_of_isIso_map _


instance (priority := 100) reflectsIsomorphisms_of_full_and_faithful
    (F : C ⥤ D) [F.Full] [F.Faithful] :
    F.ReflectsIsomorphisms :=
  (Functor.FullyFaithful.ofFullyFaithful F).reflectsIsomorphisms


instance reflectsIsomorphisms_comp (F : C ⥤ D) (G : D ⥤ E)
    [F.ReflectsIsomorphisms] [G.ReflectsIsomorphisms] :
    (F ⋙ G).ReflectsIsomorphisms :=
  ⟨fun f (hf : IsIso (G.map _)) => by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.ReflectsIsomorphisms
      inst✝ : G.ReflectsIsomorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (G.map (F.map f))
      ⊢ CategoryTheory.IsIso f
    -/
    haveI := isIso_of_reflects_iso (F.map f) G
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝² : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝¹ : F.ReflectsIsomorphisms
      inst✝ : G.ReflectsIsomorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (G.map (F.map f))
      this : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso f
    -/
    exact isIso_of_reflects_iso f F⟩
    /-
      🎉 no goals
    -/


lemma reflectsIsomorphisms_of_comp (F : C ⥤ D) (G : D ⥤ E)
    [(F ⋙ G).ReflectsIsomorphisms] : F.ReflectsIsomorphisms where
  reflects f _ := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).ReflectsIsomorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      x✝ : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso f
    -/
    rw [← isIso_iff_of_reflects_iso _ (F ⋙ G)]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).ReflectsIsomorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      x✝ : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso ((F.comp G).map f)
    -/
    dsimp
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      inst✝ : (F.comp G).ReflectsIsomorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      x✝ : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso (G.map (F.map f))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance (priority := 100) reflectsIsomorphisms_of_reflectsMonomorphisms_of_reflectsEpimorphisms
    [Balanced C] (F : C ⥤ D) [ReflectsMonomorphisms F] [ReflectsEpimorphisms F] :
    F.ReflectsIsomorphisms where
  reflects f hf := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.Balanced C
      F : CategoryTheory.Functor C D
      inst✝¹ : F.ReflectsMonomorphisms
      inst✝ : F.ReflectsEpimorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso f
    -/
    haveI : Epi f := epi_of_epi_map F inferInstance
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.Balanced C
      F : CategoryTheory.Functor C D
      inst✝¹ : F.ReflectsMonomorphisms
      inst✝ : F.ReflectsEpimorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (F.map f)
      this : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsIso f
    -/
    haveI : Mono f := mono_of_mono_map F inferInstance
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝³ : CategoryTheory.Category.{v₃, u₃} E
      inst✝² : CategoryTheory.Balanced C
      F : CategoryTheory.Functor C D
      inst✝¹ : F.ReflectsMonomorphisms
      inst✝ : F.ReflectsEpimorphisms
      A✝ B✝ : C
      f : Quiver.Hom A✝ B✝
      hf : CategoryTheory.IsIso (F.map f)
      this✝ : CategoryTheory.Epi f
      this : CategoryTheory.Mono f
      ⊢ CategoryTheory.IsIso f
    -/
    exact isIso_of_mono_of_epi f
    /-
      🎉 no goals
    -/


instance (F : D ⥤ E) [F.ReflectsIsomorphisms] :
    ((whiskeringRight C D E).obj F).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.ReflectsIsomorphisms
      X Y : CategoryTheory.Functor C D
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso (((CategoryTheory.whiskeringRight C D E).obj F).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    rw [NatTrans.isIso_iff_isIso_app]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.ReflectsIsomorphisms
      X Y : CategoryTheory.Functor C D
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso (((CategoryTheory.whiskeringRight C D E).obj F).map f)
      ⊢ ∀ (X_1 : C), CategoryTheory.IsIso (f.app X_1)
    -/
    intro Z
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.ReflectsIsomorphisms
      X Y : CategoryTheory.Functor C D
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso (((CategoryTheory.whiskeringRight C D E).obj F).map f)
      Z : C
      ⊢ CategoryTheory.IsIso (f.app Z)
    -/
    rw [← isIso_iff_of_reflects_iso _ F]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.ReflectsIsomorphisms
      X Y : CategoryTheory.Functor C D
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso (((CategoryTheory.whiskeringRight C D E).obj F).map f)
      Z : C
      ⊢ CategoryTheory.IsIso (F.map (f.app Z))
    -/
    change IsIso ((((whiskeringRight C D E).obj F).map f).app Z)
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor D E
      inst✝ : F.ReflectsIsomorphisms
      X Y : CategoryTheory.Functor C D
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso (((CategoryTheory.whiskeringRight C D E).obj F).map f)
      Z : C
      ⊢ CategoryTheory.IsIso ((((CategoryTheory.whiskeringRight C D E).obj F).map f) …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma Functor.balanced_of_preserves (F : C ⥤ D)
    [F.ReflectsIsomorphisms] [F.PreservesEpimorphisms] [F.PreservesMonomorphisms] [Balanced D] :
    Balanced C where
  isIso_of_mono_of_epi f _ _ := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.ReflectsIsomorphisms
      inst✝² : F.PreservesEpimorphisms
      inst✝¹ : F.PreservesMonomorphisms
      inst✝ : CategoryTheory.Balanced D
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝¹ : CategoryTheory.Mono f
      x✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsIso f
    -/
    rw [← isIso_iff_of_reflects_iso (F := F)]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      inst✝³ : F.ReflectsIsomorphisms
      inst✝² : F.PreservesEpimorphisms
      inst✝¹ : F.PreservesMonomorphisms
      inst✝ : CategoryTheory.Balanced D
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝¹ : CategoryTheory.Mono f
      x✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsIso (F.map f)
    -/
    exact isIso_of_mono_of_epi _
    /-
      🎉 no goals
    -/


