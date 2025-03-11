lemma IsPushout.hasLiftingProperty (h : IsPushout s f g t)
    {Z' W' : C} (g' : Z' ⟶ W') [HasLiftingProperty f g'] : HasLiftingProperty g g' where
  sq_hasLift := fun {u v} sq ↦ by
    have w : (s ≫ u) ≫ g' = f ≫ (t ≫ v) := by
      rw [← Category.assoc, ← h.w, Category.assoc, Category.assoc, sq.w]
    exact ⟨h.desc u (CommSq.mk w).lift (by rw [CommSq.fac_left]), h.inl_desc ..,
      h.hom_ext (by rw [h.inl_desc_assoc, sq.w]) (by rw [h.inr_desc_assoc, CommSq.fac_right])⟩


lemma IsPullback.hasLiftingProperty (h : IsPullback s f g t)
    {X' Y' : C} (f' : X' ⟶ Y') [HasLiftingProperty f' g] : HasLiftingProperty f' f where
  sq_hasLift := fun {u v} sq ↦ by
    have w : (u ≫ s) ≫ g = f' ≫ v ≫ t := by
      rw [Category.assoc, h.toCommSq.w, ← Category.assoc, ← Category.assoc, sq.w]
    exact ⟨h.lift (CommSq.mk w).lift v (by rw [CommSq.fac_right]),
      h.hom_ext (by rw [Category.assoc, h.lift_fst, CommSq.fac_left])
        (by rw [Category.assoc, h.lift_snd, sq.w]), h.lift_snd _ _ _⟩


instance [HasPushout s f] {T₁ T₂ : C} (p : T₁ ⟶ T₂) [HasLiftingProperty f p] :
    HasLiftingProperty (pushout.inl s f) p :=
  (IsPushout.of_hasPushout s f).hasLiftingProperty p


instance [HasPushout s f] {T₁ T₂ : C} (p : T₁ ⟶ T₂) [HasLiftingProperty s p] :
    HasLiftingProperty (pushout.inr s f) p :=
  (IsPushout.of_hasPushout s f).flip.hasLiftingProperty p


instance [HasPullback g t] {T₁ T₂ : C} (p : T₁ ⟶ T₂) [HasLiftingProperty p g] :
    HasLiftingProperty p (pullback.snd g t) :=
  (IsPullback.of_hasPullback g t).hasLiftingProperty p


instance [HasPullback g t] {T₁ T₂ : C} (p : T₁ ⟶ T₂) [HasLiftingProperty p t] :
    HasLiftingProperty p (pullback.fst g t) :=
  (IsPullback.of_hasPullback g t).flip.hasLiftingProperty p


instance {J : Type*} {A B : J → C} [HasProduct A] [HasProduct B]
    (f : (j : J) → A j ⟶ B j) {X Y : C} (p : X ⟶ Y)
    [∀ j, HasLiftingProperty p (f j)] :
    HasLiftingProperty p (Limits.Pi.map f) where
  sq_hasLift {t b} sq := by
    have sq' (j : J) :
        CommSq (t ≫ Pi.π _ j) p (f j) (b ≫ Pi.π _ j) :=
      ⟨by rw [← Category.assoc, ← sq.w]; simp⟩
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      X✝ Y✝ Z W : C
      f✝ : Quiver.Hom X✝ Y✝
      s : Quiver.Hom X✝ Z
      g : Quiver.Hom Z W
      t✝ : Quiver.Hom Y✝ W
      J : Type u_2
      A B : J → C
      inst✝² : CategoryTheory.Limits.HasProduct A
      inst✝¹ : CategoryTheory.Limits.HasProduct B
      f : (j : J) → Quiver.Hom (A j) (B j)
      X Y : C
      p : Quiver.Hom X Y
      inst✝ : ∀ (j : J), CategoryTheory.HasLiftingProperty p (f j)
      t : Quiver.Hom X (CategoryTheory.Limits.piObj A)
      b : Quiver.Hom Y (CategoryTheory.Limits.piObj B)
      sq : CategoryTheory.CommSq t p (CategoryTheory.Limits.Pi.map f) b
      sq' : ∀ (j : J), CategoryTheory.CommSq (CategoryTheory.CategoryStruct.comp t ( …
      ⊢ sq.HasLift
    -/
    exact ⟨⟨{ l := Pi.lift (fun j ↦ (sq' j).lift) }⟩⟩
    /-
      🎉 no goals
    -/


instance {J : Type*} {A B : J → C} [HasCoproduct A] [HasCoproduct B]
    (f : (j : J) → A j ⟶ B j) {X Y : C} (p : X ⟶ Y)
    [∀ j, HasLiftingProperty (f j) p] :
    HasLiftingProperty (Limits.Sigma.map f) p where
  sq_hasLift {t b} sq := by
    have sq' (j : J) :
        CommSq (Sigma.ι _ j ≫ t) (f j) p (Sigma.ι _ j ≫ b) :=
      ⟨by simp [sq.w]⟩
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      X✝ Y✝ Z W : C
      f✝ : Quiver.Hom X✝ Y✝
      s : Quiver.Hom X✝ Z
      g : Quiver.Hom Z W
      t✝ : Quiver.Hom Y✝ W
      J : Type u_2
      A B : J → C
      inst✝² : CategoryTheory.Limits.HasCoproduct A
      inst✝¹ : CategoryTheory.Limits.HasCoproduct B
      f : (j : J) → Quiver.Hom (A j) (B j)
      X Y : C
      p : Quiver.Hom X Y
      inst✝ : ∀ (j : J), CategoryTheory.HasLiftingProperty (f j) p
      t : Quiver.Hom (CategoryTheory.Limits.sigmaObj A) X
      b : Quiver.Hom (CategoryTheory.Limits.sigmaObj B) Y
      sq : CategoryTheory.CommSq t (CategoryTheory.Limits.Sigma.map f) p b
      sq' : ∀ (j : J), CategoryTheory.CommSq (CategoryTheory.CategoryStruct.comp (Ca …
      ⊢ sq.HasLift
    -/
    exact ⟨⟨{ l := Sigma.desc (fun j ↦ (sq' j).lift) }⟩⟩
    /-
      🎉 no goals
    -/


