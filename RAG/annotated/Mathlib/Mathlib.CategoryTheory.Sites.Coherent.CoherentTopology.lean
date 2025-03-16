/--
For a precoherent category, any sieve that contains an `EffectiveEpiFamily` is a sieve of the
coherent topology.
Note: This is one direction of `mem_sieves_iff_hasEffectiveEpiFamily`, but is needed for the proof.
-/
theorem coherentTopology.mem_sieves_of_hasEffectiveEpiFamily (S : Sieve X) :
    (∃ (α : Type) (_ : Finite α) (Y : α → C) (π : (a : α) → (Y a ⟶ X)),
      EffectiveEpiFamily Y π ∧ (∀ a : α, (S.arrows) (π a)) ) →
        (S ∈ (coherentTopology C) X) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ (Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Catego …
  -/
  intro ⟨α, _, Y, π, hπ⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    X : C
    S : CategoryTheory.Sieve X
    α : Type
    w✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    hπ : And (CategoryTheory.EffectiveEpiFamily Y π) (∀ (a : α), S.arrows (π a))
    ⊢ Membership.mem ((CategoryTheory.coherentTopology C) X) S
  -/
  apply (coherentCoverage C).mem_toGrothendieck_sieves_of_superset (R := Presieve.ofArrows Y π)
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      X : C
      S : CategoryTheory.Sieve X
      α : Type
      w✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      hπ : And (CategoryTheory.EffectiveEpiFamily Y π) (∀ (a : α), S.arrows (π a))
      ⊢ LE.le (CategoryTheory.Presieve.ofArrows Y π) S.arrows
    -/
  · exact fun _ _ h ↦ by cases h; exact hπ.2 _
    /-
      🎉 no goals
    -/
    /-
      case hR
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      X : C
      S : CategoryTheory.Sieve X
      α : Type
      w✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      hπ : And (CategoryTheory.EffectiveEpiFamily Y π) (∀ (a : α), S.arrows (π a))
      ⊢ Membership.mem ((CategoryTheory.coherentCoverage C).covering X) (CategoryThe …
    -/
  · exact ⟨_, inferInstance, Y, π, rfl, hπ.1⟩
    /-
      🎉 no goals
    -/


/--
Effective epi families in a precoherent category are transitive, in the sense that an
`EffectiveEpiFamily` and an `EffectiveEpiFamily` over each member, the composition is an
`EffectiveEpiFamily`.
Note: The finiteness condition is an artifact of the proof and is probably unnecessary.
-/
theorem EffectiveEpiFamily.transitive_of_finite {α : Type} [Finite α] {Y : α → C}
    (π : (a : α) → (Y a ⟶ X)) (h : EffectiveEpiFamily Y π) {β : α → Type} [∀ (a : α), Finite (β a)]
    {Y_n : (a : α) → β a → C} (π_n : (a : α) → (b : β a) → (Y_n a b ⟶ Y a))
    (H : ∀ a, EffectiveEpiFamily (Y_n a) (π_n a)) :
    EffectiveEpiFamily
      (fun (c : Σ a, β a) => Y_n c.fst c.snd) (fun c => π_n c.fst c.snd ≫ π c.fst) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Precoherent C
    X : C
    α : Type
    inst✝¹ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : CategoryTheory.EffectiveEpiFamily Y π
    β : α → Type
    inst✝ : ∀ (a : α), Finite (β a)
    Y_n : (a : α) → β a → C
    π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
    H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
    ⊢ CategoryTheory.EffectiveEpiFamily (fun c => Y_n c.fst c.snd) fun c => Catego …
  -/
  rw [← Sieve.effectiveEpimorphic_family]
  suffices h₂ : (Sieve.generate (Presieve.ofArrows (fun (⟨a, b⟩ : Σ _, β _) => Y_n a b)
        (fun ⟨a,b⟩ => π_n a b ≫ π a))) ∈ (coherentTopology C) X by
    change Nonempty _
    rw [← Sieve.forallYonedaIsSheaf_iff_colimit]
    exact fun W => coherentTopology.isSheaf_yoneda_obj W _ h₂
  -- Show that a covering sieve is a colimit, which implies the original set of arrows is regular
  -- epimorphic. We use the transitivity property of saturation
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Precoherent C
    X : C
    α : Type
    inst✝¹ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    h : CategoryTheory.EffectiveEpiFamily Y π
    β : α → Type
    inst✝ : ∀ (a : α), Finite (β a)
    Y_n : (a : α) → β a → C
    π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
    H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
    ⊢ Membership.mem ((CategoryTheory.coherentTopology C) X) (CategoryTheory.Sieve …
  -/
  apply Coverage.Saturate.transitive X (Sieve.generate (Presieve.ofArrows Y π))
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      ⊢ (CategoryTheory.coherentCoverage C).Saturate X (CategoryTheory.Sieve.generat …
    -/
  · apply Coverage.Saturate.of
    /-
      case a.hS
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      ⊢ Membership.mem ((CategoryTheory.coherentCoverage C).covering X) (CategoryThe …
    -/
    use α, inferInstance, Y, π
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 X⦄, (CategoryTheory.Sieve.generate (Category …
    -/
  · intro V f ⟨Y₁, h, g, ⟨hY, hf⟩⟩
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h✝ : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows Y π g
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ (CategoryTheory.coherentCoverage C).Saturate V (CategoryTheory.Sieve.pullbac …
    -/
    rw [← hf, Sieve.pullback_comp]
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h✝ : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows Y π g
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ (CategoryTheory.coherentCoverage C).Saturate V (CategoryTheory.Sieve.pullbac …
    -/
    apply (coherentTopology C).pullback_stable'
    /-
      case a.a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h✝ : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows Y π g
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ Membership.mem ((CategoryTheory.coherentTopology C).sieves Y₁) (CategoryTheo …
    -/
    apply coherentTopology.mem_sieves_of_hasEffectiveEpiFamily
    -- Need to show that the pullback of the family `π_n` to a given `Y i` is effective epimorphic
    /-
      case a.a.a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Precoherent C
      X : C
      α : Type
      inst✝¹ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h✝ : CategoryTheory.EffectiveEpiFamily Y π
      β : α → Type
      inst✝ : ∀ (a : α), Finite (β a)
      Y_n : (a : α) → β a → C
      π_n : (a : α) → (b : β a) → Quiver.Hom (Y_n a b) (Y a)
      H : ∀ (a : α), CategoryTheory.EffectiveEpiFamily (Y_n a) (π_n a)
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows Y π g
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ Exists fun α_1 => Exists fun x => Exists fun Y_1 => Exists fun π_1 => And (C …
    -/
    obtain ⟨i⟩ := hY
    exact ⟨β i, inferInstance, Y_n i, π_n i, H i, fun b ↦
      ⟨Y_n i b, (𝟙 _), π_n i b ≫ π i, ⟨(⟨i, b⟩ : Σ (i : α), β i)⟩, by simp⟩⟩


instance precoherentEffectiveEpiFamilyCompEffectiveEpis
    {α : Type} [Finite α] {Y Z : α → C} (π : (a : α) → (Y a ⟶ X)) [EffectiveEpiFamily Y π]
    (f : (a : α) → Z a ⟶ Y a) [h : ∀ a, EffectiveEpi (f a)] :
    EffectiveEpiFamily _ fun a ↦ f a ≫ π a := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Precoherent C
    X : C
    α : Type
    inst✝¹ : Finite α
    Y Z : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    inst✝ : CategoryTheory.EffectiveEpiFamily Y π
    f : (a : α) → Quiver.Hom (Z a) (Y a)
    h : ∀ (a : α), CategoryTheory.EffectiveEpi (f a)
    ⊢ CategoryTheory.EffectiveEpiFamily Z fun a => CategoryTheory.CategoryStruct.c …
  -/
  simp_rw [effectiveEpi_iff_effectiveEpiFamily] at h
  exact EffectiveEpiFamily.reindex (e := Equiv.sigmaPUnit α) _ _
    (EffectiveEpiFamily.transitive_of_finite (β := fun _ ↦ Unit) _ inferInstance _ h)


/--
A sieve belongs to the coherent topology if and only if it contains a finite
`EffectiveEpiFamily`.
-/
theorem coherentTopology.mem_sieves_iff_hasEffectiveEpiFamily (S : Sieve X) :
    (S ∈ (coherentTopology C) X) ↔
    (∃ (α : Type) (_ : Finite α) (Y : α → C) (π : (a : α) → (Y a ⟶ X)),
        EffectiveEpiFamily Y π ∧ (∀ a : α, (S.arrows) (π a)) )  := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.coherentTopology C) X) S) (Exists fun α …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.coherentTopology C) X) S → Exists fun α => E …
    -/
  · intro h
    induction h with
    | of Y T hS =>
      obtain ⟨a, h, Y', π, h', _⟩ := hS
      refine ⟨a, h, Y', π, inferInstance, fun a' ↦ ?_⟩
      obtain ⟨rfl, _⟩ := h'
      exact ⟨Y' a', 𝟙 Y' a', π a', Presieve.ofArrows.mk a', by simp⟩
    | top Y =>
      exact ⟨Unit, inferInstance, fun _ => Y, fun _ => (𝟙 Y), inferInstance, by simp⟩
    | transitive Y R S _ _ a b =>
      obtain ⟨α, w, Y₁, π, ⟨h₁,h₂⟩⟩ := a
      choose β _ Y_n π_n H using fun a => b (h₂ a)
      exact ⟨(Σ a, β a), inferInstance, fun ⟨a,b⟩ => Y_n a b, fun ⟨a, b⟩ => (π_n a b) ≫ (π a),
        EffectiveEpiFamily.transitive_of_finite _ h₁ _ (fun a => (H a).1),
        fun c => (H c.fst).2 c.snd⟩
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Catego …
    -/
  · exact coherentTopology.mem_sieves_of_hasEffectiveEpiFamily S
    /-
      🎉 no goals
    -/


