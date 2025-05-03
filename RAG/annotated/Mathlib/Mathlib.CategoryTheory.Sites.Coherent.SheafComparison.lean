instance : F.IsCoverDense (coherentTopology _) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    ⊢ F.IsCoverDense (CategoryTheory.coherentTopology D)
  -/
  refine F.isCoverDense_of_generate_singleton_functor_π_mem _ fun B ↦ ⟨_, F.effectiveEpiOver B, ?_⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    B : D
    ⊢ Membership.mem ((CategoryTheory.coherentTopology D) B) (CategoryTheory.Sieve …
  -/
  apply Coverage.Saturate.of
  refine ⟨Unit, inferInstance, fun _ => F.effectiveEpiOverObj B,
    fun _ => F.effectiveEpiOver B, ?_ , ?_⟩
    /-
      case hS.refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B : D
      ⊢ Eq (CategoryTheory.Presieve.singleton (F.effectiveEpiOver B)) (CategoryTheor …
    -/
  · funext; ext -- Do we want `Presieve.ext`?
    /-
      case hS.refine_1.h.h.a
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B x✝¹ : D
      x✝ : Quiver.Hom x✝¹ B
      ⊢ Iff (CategoryTheory.Presieve.singleton (F.effectiveEpiOver B) x✝) (CategoryT …
    -/
    refine ⟨fun ⟨⟩ ↦ ⟨()⟩, ?_⟩
    /-
      case hS.refine_1.h.h.a
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B x✝¹ : D
      x✝ : Quiver.Hom x✝¹ B
      ⊢ CategoryTheory.Presieve.ofArrows (fun x => F.effectiveEpiOverObj B) (fun x = …
    -/
    rintro ⟨⟩
    /-
      case hS.refine_1.h.h.a.mk
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B Y : D
      i✝ : Unit
      ⊢ CategoryTheory.Presieve.singleton (F.effectiveEpiOver B) (F.effectiveEpiOver …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case hS.refine_2
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B : D
      ⊢ CategoryTheory.EffectiveEpiFamily (fun x => F.effectiveEpiOverObj B) fun x = …
    -/
  · rw [← effectiveEpi_iff_effectiveEpiFamily]
    /-
      case hS.refine_2
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      B : D
      ⊢ CategoryTheory.EffectiveEpi (F.effectiveEpiOver B)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem exists_effectiveEpiFamily_iff_mem_induced (X : C) (S : Sieve X) :
    (∃ (α : Type) (_ : Finite α) (Y : α → C) (π : (a : α) → (Y a ⟶ X)),
      EffectiveEpiFamily Y π ∧ (∀ a : α, (S.arrows) (π a)) ) ↔
    (S ∈ F.inducedTopology (coherentTopology _) X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Ca …
  -/
  refine ⟨fun ⟨α, _, Y, π, ⟨H₁, H₂⟩⟩ ↦ ?_, fun hS ↦ ?_⟩
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      x✝ : Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Cate …
      α : Type
      w✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), S.arrows (π a)
      ⊢ Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X) S
    -/
  · apply (mem_sieves_iff_hasEffectiveEpiFamily (Sieve.functorPushforward _ S)).mpr
    refine ⟨α, inferInstance, fun i => F.obj (Y i),
      fun i => F.map (π i), ⟨?_,
      fun a => Sieve.image_mem_functorPushforward F S (H₂ a)⟩⟩
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      x✝ : Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Cate …
      α : Type
      w✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), S.arrows (π a)
      ⊢ CategoryTheory.EffectiveEpiFamily (fun i => F.obj (Y i)) fun i => F.map (π i)
    -/
    exact F.map_finite_effectiveEpiFamily _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      ⊢ Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Categor …
    -/
  · obtain ⟨α, _, Y, π, ⟨H₁, H₂⟩⟩ := (mem_sieves_iff_hasEffectiveEpiFamily _).mp hS
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      α : Type
      w✝ : Finite α
      Y : α → D
      π : (a : α) → Quiver.Hom (Y a) (F.obj X)
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
      ⊢ Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Categor …
    -/
    refine ⟨α, inferInstance, ?_⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      α : Type
      w✝ : Finite α
      Y : α → D
      π : (a : α) → Quiver.Hom (Y a) (F.obj X)
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpiFamily Y π)  …
    -/
    let Z : α → C := fun a ↦ (Functor.EffectivelyEnough.presentation (F := F) (Y a)).some.p
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      α : Type
      w✝ : Finite α
      Y : α → D
      π : (a : α) → Quiver.Hom (Y a) (F.obj X)
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
      Z : α → C := fun a => ⋯.some.p
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpiFamily Y π)  …
    -/
    let g₀ : (a : α) → F.obj (Z a) ⟶ Y a := fun a ↦ F.effectiveEpiOver (Y a)
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      α : Type
      w✝ : Finite α
      Y : α → D
      π : (a : α) → Quiver.Hom (Y a) (F.obj X)
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
      Z : α → C := fun a => ⋯.some.p
      g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpiFamily Y π)  …
    -/
    have : EffectiveEpiFamily _ (fun a ↦ g₀ a ≫ π a) := inferInstance
    /-
      case refine_2.intro.intro.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
      α : Type
      w✝ : Finite α
      Y : α → D
      π : (a : α) → Quiver.Hom (Y a) (F.obj X)
      H₁ : CategoryTheory.EffectiveEpiFamily Y π
      H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
      Z : α → C := fun a => ⋯.some.p
      g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
      this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpiFamily Y π)  …
    -/
    refine ⟨Z , fun a ↦ F.preimage (g₀ a ≫ π a), ?_, fun a ↦ (?_ : S.arrows (F.preimage _))⟩
      /-
        case refine_2.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        ⊢ CategoryTheory.EffectiveEpiFamily Z fun a => F.preimage (CategoryTheory.Cate …
      -/
    · refine F.finite_effectiveEpiFamily_of_map _ _ ?_
      /-
        case refine_2.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        ⊢ CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => F.map (F.p …
      -/
      simpa using this
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        a : α
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp (g₀ a) (π a)))
      -/
    · obtain ⟨W, g₁, g₂, h₁, h₂⟩ := H₂ a
      /-
        case refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.intro.intro
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        a : α
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom (Y a) (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq (π a) (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp (g₀ a) (π a)))
      -/
      rw [h₂]
      /-
        case refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.intro.intro
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        a : α
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom (Y a) (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq (π a) (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp (g₀ a) (CategoryThe …
      -/
      convert S.downward_closed h₁ (F.preimage (g₀ a ≫ g₂))
      /-
        case h.e'_6
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
        inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Precoherent D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.coherentTopology D)) X …
        α : Type
        w✝ : Finite α
        Y : α → D
        π : (a : α) → Quiver.Hom (Y a) (F.obj X)
        H₁ : CategoryTheory.EffectiveEpiFamily Y π
        H₂ : ∀ (a : α), (CategoryTheory.Sieve.functorPushforward F S).arrows (π a)
        Z : α → C := fun a => ⋯.some.p
        g₀ : (a : α) → Quiver.Hom (F.obj (Z a)) (Y a) := fun a => F.effectiveEpiOver ( …
        this : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (Z a)) fun a => Categ …
        a : α
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom (Y a) (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq (π a) (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ Eq (F.preimage (CategoryTheory.CategoryStruct.comp (g₀ a) (CategoryTheory.Ca …
      -/
      exact F.map_injective (by simp)
      /-
        🎉 no goals
      -/


lemma eq_induced : haveI := F.reflects_precoherent
    coherentTopology C =
      F.inducedTopology (coherentTopology _) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    ⊢ Eq (CategoryTheory.coherentTopology C) (F.inducedTopology (CategoryTheory.co …
  -/
  ext X S
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.coherentTopology C) X) S) (Membership.m …
  -/
  have := F.reflects_precoherent
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    X : C
    S : CategoryTheory.Sieve X
    this : CategoryTheory.Precoherent C
    ⊢ Iff (Membership.mem ((CategoryTheory.coherentTopology C) X) S) (Membership.m …
  -/
  rw [← exists_effectiveEpiFamily_iff_mem_induced F X]
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
    inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Precoherent D
    X : C
    S : CategoryTheory.Sieve X
    this : CategoryTheory.Precoherent C
    ⊢ Iff (Membership.mem ((CategoryTheory.coherentTopology C) X) S) (Exists fun α …
  -/
  rw [← coherentTopology.mem_sieves_iff_hasEffectiveEpiFamily S]
  /-
    🎉 no goals
  -/


instance : haveI := F.reflects_precoherent;
    F.IsDenseSubsite (coherentTopology C) (coherentTopology D) where
  functorPushforward_mem_iff := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      ⊢ ∀ {X : C} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    rw [eq_induced F]
    #adaptation_note
    /--
    This proof used to be `rfl`,
    but has been temporarily broken by https://github.com/leanprover/lean4/pull/5329.
    It can hopefully be restored after https://github.com/leanprover/lean4/pull/5359
    -/
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesFiniteEffectiveEpiFamilies
      inst✝⁴ : F.ReflectsFiniteEffectiveEpiFamilies
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Precoherent D
      ⊢ ∀ {X : C} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/


lemma coverPreserving : haveI := F.reflects_precoherent
    CoverPreserving (coherentTopology _) (coherentTopology _) F :=
  IsDenseSubsite.coverPreserving _ _ _


/--
The equivalence from coherent sheaves on `C` to coherent sheaves on `D`, given a fully faithful
functor `F : C ⥤ D` to a precoherent category, which preserves and reflects effective epimorphic
families, and satisfies `F.EffectivelyEnough`.
-/
noncomputable
def equivalence (A : Type u₃) [Category.{v₃} A] [∀ X, HasLimitsOfShape (StructuredArrow X F.op) A] :
    haveI := F.reflects_precoherent
    Sheaf (coherentTopology C) A ≌ Sheaf (coherentTopology D) A :=
  Functor.IsDenseSubsite.sheafEquiv F _ _ _


/--
The equivalence from coherent sheaves on `C` to coherent sheaves on `D`, given a fully faithful
functor `F : C ⥤ D` to an extensive preregular category, which preserves and reflects effective
epimorphisms and satisfies `F.EffectivelyEnough`.
-/
noncomputable
def equivalence' (A : Type u₃) [Category.{v₃} A]
    [∀ X, HasLimitsOfShape (StructuredArrow X F.op) A] :
    haveI := F.reflects_precoherent
    Sheaf (coherentTopology C) A ≌ Sheaf (coherentTopology D) A :=
  Functor.IsDenseSubsite.sheafEquiv F _ _ _


instance : F.IsCoverDense (regularTopology _) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    ⊢ F.IsCoverDense (CategoryTheory.regularTopology D)
  -/
  refine F.isCoverDense_of_generate_singleton_functor_π_mem _ fun B ↦ ⟨_, F.effectiveEpiOver B, ?_⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B : D
    ⊢ Membership.mem ((CategoryTheory.regularTopology D) B) (CategoryTheory.Sieve. …
  -/
  apply Coverage.Saturate.of
  /-
    case hS
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B : D
    ⊢ Membership.mem ((CategoryTheory.regularCoverage D).covering B) (CategoryTheo …
  -/
  refine ⟨F.effectiveEpiOverObj B, F.effectiveEpiOver B, ?_, inferInstance⟩
  /-
    case hS
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B : D
    ⊢ Eq (CategoryTheory.Presieve.singleton (F.effectiveEpiOver B)) (CategoryTheor …
  -/
  funext; ext -- Do we want `Presieve.ext`?
  /-
    case hS.h.h.a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B x✝¹ : D
    x✝ : Quiver.Hom x✝¹ B
    ⊢ Iff (CategoryTheory.Presieve.singleton (F.effectiveEpiOver B) x✝) (CategoryT …
  -/
  refine ⟨fun ⟨⟩ ↦ ⟨()⟩, ?_⟩
  /-
    case hS.h.h.a
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B x✝¹ : D
    x✝ : Quiver.Hom x✝¹ B
    ⊢ CategoryTheory.Presieve.ofArrows (fun x => F.effectiveEpiOverObj B) (fun x = …
  -/
  rintro ⟨⟩
  /-
    case hS.h.h.a.mk
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    B Y : D
    i✝ : Unit
    ⊢ CategoryTheory.Presieve.singleton (F.effectiveEpiOver B) (F.effectiveEpiOver …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem exists_effectiveEpi_iff_mem_induced (X : C) (S : Sieve X) :
    (∃ (Y : C) (π : Y ⟶ X),
      EffectiveEpi π ∧ S.arrows π) ↔
    (S ∈ F.inducedTopology (regularTopology _) X) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S. …
  -/
  refine ⟨fun ⟨Y, π, ⟨H₁, H₂⟩⟩ ↦ ?_, fun hS ↦ ?_⟩
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      x✝ : Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.ar …
      Y : C
      π : Quiver.Hom Y X
      H₁ : CategoryTheory.EffectiveEpi π
      H₂ : S.arrows π
      ⊢ Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
    -/
  · apply (mem_sieves_iff_hasEffectiveEpi (Sieve.functorPushforward _ S)).mpr
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      x✝ : Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.ar …
      Y : C
      π : Quiver.Hom Y X
      H₁ : CategoryTheory.EffectiveEpi π
      H₂ : S.arrows π
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) ((Catego …
    -/
    refine ⟨F.obj Y, F.map π, ⟨?_, Sieve.image_mem_functorPushforward F S H₂⟩⟩
    /-
      case refine_1
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      x✝ : Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.ar …
      Y : C
      π : Quiver.Hom Y X
      H₁ : CategoryTheory.EffectiveEpi π
      H₂ : S.arrows π
      ⊢ CategoryTheory.EffectiveEpi (F.map π)
    -/
    exact F.map_effectiveEpi _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.arrow …
    -/
  · obtain ⟨Y, π, ⟨H₁, H₂⟩⟩ := (mem_sieves_iff_hasEffectiveEpi _).mp hS
    /-
      case refine_2.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
      Y : D
      π : Quiver.Hom Y (F.obj X)
      H₁ : CategoryTheory.EffectiveEpi π
      H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.arrow …
    -/
    let g₀ := F.effectiveEpiOver Y
    /-
      case refine_2.intro.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
      Y : D
      π : Quiver.Hom Y (F.obj X)
      H₁ : CategoryTheory.EffectiveEpi π
      H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
      g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
      ⊢ Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.arrow …
    -/
    refine ⟨_, F.preimage (g₀ ≫ π), ?_, (?_ : S.arrows (F.preimage _))⟩
      /-
        case refine_2.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        ⊢ CategoryTheory.EffectiveEpi (F.preimage (CategoryTheory.CategoryStruct.comp  …
      -/
    · refine F.effectiveEpi_of_map _ ?_
      /-
        case refine_2.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        ⊢ CategoryTheory.EffectiveEpi (F.map (F.preimage (CategoryTheory.CategoryStruc …
      -/
      simp only [map_preimage]
      /-
        case refine_2.intro.intro.intro.refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.CategoryStruct.comp g₀ π)
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.intro.refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        H₂ : (CategoryTheory.Sieve.functorPushforward F S).arrows π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp g₀ π))
      -/
    · obtain ⟨W, g₁, g₂, h₁, h₂⟩ := H₂
      /-
        case refine_2.intro.intro.intro.refine_2.intro.intro.intro.intro
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom Y (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq π (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp g₀ π))
      -/
      rw [h₂]
      /-
        case refine_2.intro.intro.intro.refine_2.intro.intro.intro.intro
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom Y (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq π (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ S.arrows (F.preimage (CategoryTheory.CategoryStruct.comp g₀ (CategoryTheory. …
      -/
      convert S.downward_closed h₁ (F.preimage (g₀ ≫ g₂))
      /-
        case h.e'_6
        C : Type u_1
        D : Type u_2
        inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
        inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
        F : CategoryTheory.Functor C D
        inst✝⁵ : F.PreservesEffectiveEpis
        inst✝⁴ : F.ReflectsEffectiveEpis
        inst✝³ : F.Full
        inst✝² : F.Faithful
        inst✝¹ : F.EffectivelyEnough
        inst✝ : CategoryTheory.Preregular D
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem ((F.inducedTopology (CategoryTheory.regularTopology D)) X) S
        Y : D
        π : Quiver.Hom Y (F.obj X)
        H₁ : CategoryTheory.EffectiveEpi π
        g₀ : Quiver.Hom (F.effectiveEpiOverObj Y) Y := F.effectiveEpiOver Y
        W : C
        g₁ : Quiver.Hom W X
        g₂ : Quiver.Hom Y (F.obj W)
        h₁ : S.arrows g₁
        h₂ : Eq π (CategoryTheory.CategoryStruct.comp g₂ (F.map g₁))
        ⊢ Eq (F.preimage (CategoryTheory.CategoryStruct.comp g₀ (CategoryTheory.Catego …
      -/
      exact F.map_injective (by simp)
      /-
        🎉 no goals
      -/


lemma eq_induced : haveI := F.reflects_preregular
    regularTopology C =
      F.inducedTopology (regularTopology _) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    ⊢ Eq (CategoryTheory.regularTopology C) (F.inducedTopology (CategoryTheory.reg …
  -/
  ext X S
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.regularTopology C) X) S) (Membership.me …
  -/
  have := F.reflects_preregular
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    X : C
    S : CategoryTheory.Sieve X
    this : CategoryTheory.Preregular C
    ⊢ Iff (Membership.mem ((CategoryTheory.regularTopology C) X) S) (Membership.me …
  -/
  rw [← exists_effectiveEpi_iff_mem_induced F X]
  /-
    case h.h.h
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    inst✝⁵ : F.PreservesEffectiveEpis
    inst✝⁴ : F.ReflectsEffectiveEpis
    inst✝³ : F.Full
    inst✝² : F.Faithful
    inst✝¹ : F.EffectivelyEnough
    inst✝ : CategoryTheory.Preregular D
    X : C
    S : CategoryTheory.Sieve X
    this : CategoryTheory.Preregular C
    ⊢ Iff (Membership.mem ((CategoryTheory.regularTopology C) X) S) (Exists fun Y  …
  -/
  rw [← mem_sieves_iff_hasEffectiveEpi S]
  /-
    🎉 no goals
  -/


instance : haveI := F.reflects_preregular;
    F.IsDenseSubsite (regularTopology C) (regularTopology D) where
  functorPushforward_mem_iff := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      ⊢ ∀ {X : C} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    rw [eq_induced F]
    #adaptation_note
    /--
    This proof used to be `rfl`,
    but has been temporarily broken by https://github.com/leanprover/lean4/pull/5329.
    It can hopefully be restored after https://github.com/leanprover/lean4/pull/5359
    -/
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : F.PreservesEffectiveEpis
      inst✝⁴ : F.ReflectsEffectiveEpis
      inst✝³ : F.Full
      inst✝² : F.Faithful
      inst✝¹ : F.EffectivelyEnough
      inst✝ : CategoryTheory.Preregular D
      ⊢ ∀ {X : C} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/


lemma coverPreserving : haveI := F.reflects_preregular
    CoverPreserving (regularTopology _) (regularTopology _) F :=
  IsDenseSubsite.coverPreserving _ _ _


/--
The equivalence from regular sheaves on `C` to regular sheaves on `D`, given a fully faithful
functor `F : C ⥤ D` to a preregular category, which preserves and reflects effective
epimorphisms and satisfies `F.EffectivelyEnough`.
-/
noncomputable
def equivalence (A : Type u₃) [Category.{v₃} A] [∀ X, HasLimitsOfShape (StructuredArrow X F.op) A] :
    haveI := F.reflects_preregular
    Sheaf (regularTopology C) A ≌ Sheaf (regularTopology D) A :=
  Functor.IsDenseSubsite.sheafEquiv F _ _ _


theorem isSheaf_coherent_iff_regular_and_extensive [Preregular C] [FinitaryPreExtensive C] :
    IsSheaf (coherentTopology C) F ↔
    IsSheaf (extensiveTopology C) F ∧ IsSheaf (regularTopology C) F := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F)  …
  -/
  rw [← extensive_regular_generate_coherent]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.Coverage.toGrothendieck …
  -/
  exact isSheaf_sup (extensiveCoverage C) (regularCoverage C) F
  /-
    🎉 no goals
  -/


theorem isSheaf_iff_preservesFiniteProducts_and_equalizerCondition
    [Preregular C] [FinitaryExtensive C]
    [h : ∀ {Y X : C} (f : Y ⟶ X) [EffectiveEpi f], HasPullback f f] :
    IsSheaf (coherentTopology C) F ↔ PreservesFiniteProducts F ∧
      EqualizerCondition F := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F)  …
  -/
  rw [isSheaf_coherent_iff_regular_and_extensive]
  exact and_congr (isSheaf_iff_preservesFiniteProducts _)
    (@equalizerCondition_iff_isSheaf _ _ _ _ F _ h).symm


noncomputable instance [Preregular C] [FinitaryExtensive C]
    (F : Sheaf (coherentTopology C) A) : PreservesFiniteProducts F.val :=
  (Presheaf.isSheaf_iff_preservesFiniteProducts F.val).1
    ((Presheaf.isSheaf_coherent_iff_regular_and_extensive F.val).mp F.cond).1


theorem isSheaf_iff_preservesFiniteProducts_of_projective [Preregular C] [FinitaryExtensive C]
    [∀ (X : C), Projective X] :
    IsSheaf (coherentTopology C) F ↔ PreservesFiniteProducts F := by
  rw [isSheaf_coherent_iff_regular_and_extensive, and_iff_left (isSheaf_of_projective F),
    isSheaf_iff_preservesFiniteProducts]


theorem isSheaf_iff_extensiveSheaf_of_projective [Preregular C] [FinitaryExtensive C]
    [∀ (X : C), Projective X] :
    IsSheaf (coherentTopology C) F ↔ IsSheaf (extensiveTopology C) F := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝³ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    inst✝ : ∀ (X : C), CategoryTheory.Projective X
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F)  …
  -/
  rw [isSheaf_iff_preservesFiniteProducts_of_projective, isSheaf_iff_preservesFiniteProducts]
  /-
    🎉 no goals
  -/


/--
The categories of coherent sheaves and extensive sheaves on `C` are equivalent if `C` is
preregular, finitary extensive, and every object is projective.
-/
@[simps]
def coherentExtensiveEquivalence [Preregular C] [FinitaryExtensive C] [∀ (X : C), Projective X] :
    Sheaf (coherentTopology C) A ≌ Sheaf (extensiveTopology C) A where
  functor := {
    obj := fun F ↦ ⟨F.val, (isSheaf_iff_extensiveSheaf_of_projective F.val).mp F.cond⟩
    map := fun f ↦ ⟨f.val⟩ }
  inverse := {
    obj := fun F ↦ ⟨F.val, (isSheaf_iff_extensiveSheaf_of_projective F.val).mpr F.cond⟩
    map := fun f ↦ ⟨f.val⟩ }
  unitIso := Iso.refl _
  counitIso := Iso.refl _


lemma isSheaf_coherent_of_hasPullbacks_comp [Preregular C] [FinitaryExtensive C]
    [h : ∀ {Y X : C} (f : Y ⟶ X) [EffectiveEpi f], HasPullback f f] [PreservesFiniteLimits s]
    (hF : IsSheaf (coherentTopology C) F) : IsSheaf (coherentTopology C) (F ⋙ s) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits s
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) (F.comp s)
  -/
  rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition (h := h)] at hF ⊢
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits s
    hF : And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.reg …
    ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)) (CategoryTheo …
  -/
  have := hF.1
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits s
    hF : And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.reg …
    this : CategoryTheory.Limits.PreservesFiniteProducts F
    ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)) (CategoryTheo …
  -/
  refine ⟨inferInstance, fun _ _ π _ c hc ↦ ⟨?_⟩⟩
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits s
    hF : And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.reg …
    this : CategoryTheory.Limits.PreservesFiniteProducts F
    x✝² x✝¹ : C
    π : Quiver.Hom x✝² x✝¹
    x✝ : CategoryTheory.EffectiveEpi π
    c : CategoryTheory.Limits.PullbackCone π π
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι ((F.comp s).ma …
  -/
  exact isLimitForkMapOfIsLimit s _ (hF.2 π c hc).some
  /-
    🎉 no goals
  -/


lemma isSheaf_coherent_of_hasPullbacks_of_comp [Preregular C] [FinitaryExtensive C]
    [h : ∀ {Y X : C} (f : Y ⟶ X) [EffectiveEpi f], HasPullback f f]
    [ReflectsFiniteLimits s]
    (hF : IsSheaf (coherentTopology C) (F ⋙ s)) : IsSheaf (coherentTopology C) F := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.ReflectsFiniteLimits s
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) (F.co …
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F
  -/
  rw [isSheaf_iff_preservesFiniteProducts_and_equalizerCondition (h := h)] at hF ⊢
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.ReflectsFiniteLimits s
    hF : And (CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)) (CategoryT …
    ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.regula …
  -/
  obtain ⟨_, hF₂⟩ := hF
  /-
    case intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝³ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.Preregular C
    inst✝¹ : CategoryTheory.FinitaryExtensive C
    h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
    inst✝ : CategoryTheory.Limits.ReflectsFiniteLimits s
    left✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)
    hF₂ : CategoryTheory.regularTopology.EqualizerCondition (F.comp s)
    ⊢ And (CategoryTheory.Limits.PreservesFiniteProducts F) (CategoryTheory.regula …
  -/
  refine ⟨⟨fun J _ ↦ ⟨fun {K} ↦ ⟨fun {c} hc ↦ ?_⟩⟩⟩, fun _ _ π _ c hc ↦ ⟨?_⟩⟩
    /-
      case intro.refine_1
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      F : CategoryTheory.Functor (Opposite C) A
      B : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} B
      s : CategoryTheory.Functor A B
      inst✝² : CategoryTheory.Preregular C
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
      inst✝ : CategoryTheory.Limits.ReflectsFiniteLimits s
      left✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)
      hF₂ : CategoryTheory.regularTopology.EqualizerCondition (F.comp s)
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
    -/
  · exact ⟨isLimitOfReflects s (isLimitOfPreserves (F ⋙ s) hc)⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      F : CategoryTheory.Functor (Opposite C) A
      B : Type u₄
      inst✝³ : CategoryTheory.Category.{v₄, u₄} B
      s : CategoryTheory.Functor A B
      inst✝² : CategoryTheory.Preregular C
      inst✝¹ : CategoryTheory.FinitaryExtensive C
      h : ∀ {Y X : C} (f : Quiver.Hom Y X) [inst : CategoryTheory.EffectiveEpi f], C …
      inst✝ : CategoryTheory.Limits.ReflectsFiniteLimits s
      left✝ : CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)
      hF₂ : CategoryTheory.regularTopology.EqualizerCondition (F.comp s)
      x✝² x✝¹ : C
      π : Quiver.Hom x✝² x✝¹
      x✝ : CategoryTheory.EffectiveEpi π
      c : CategoryTheory.Limits.PullbackCone π π
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι (F.map π.op) ⋯)
    -/
  · exact isLimitOfIsLimitForkMap s _ (hF₂ π c hc).some
    /-
      🎉 no goals
    -/


lemma isSheaf_coherent_of_projective_comp [Preregular C] [FinitaryExtensive C]
    [∀ (X : C), Projective X] [PreservesFiniteProducts s]
    (hF : IsSheaf (coherentTopology C) F) : IsSheaf (coherentTopology C) (F ⋙ s) := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝³ : CategoryTheory.Preregular C
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : ∀ (X : C), CategoryTheory.Projective X
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts s
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) (F.comp s)
  -/
  rw [isSheaf_iff_preservesFiniteProducts_of_projective] at hF ⊢
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝³ : CategoryTheory.Preregular C
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : ∀ (X : C), CategoryTheory.Projective X
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts s
    hF : CategoryTheory.Limits.PreservesFiniteProducts F
    ⊢ CategoryTheory.Limits.PreservesFiniteProducts (F.comp s)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isSheaf_coherent_of_projective_of_comp [Preregular C] [FinitaryExtensive C]
    [∀ (X : C), Projective X]
    [ReflectsFiniteProducts s]
    (hF : IsSheaf (coherentTopology C) (F ⋙ s)) : IsSheaf (coherentTopology C) F := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    F : CategoryTheory.Functor (Opposite C) A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    s : CategoryTheory.Functor A B
    inst✝³ : CategoryTheory.Preregular C
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : ∀ (X : C), CategoryTheory.Projective X
    inst✝ : CategoryTheory.Limits.ReflectsFiniteProducts s
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) (F.co …
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F
  -/
  rw [isSheaf_iff_preservesFiniteProducts_of_projective] at hF ⊢
  exact ⟨fun J _ ↦ ⟨fun {K} ↦ ⟨fun {c} hc ↦
    ⟨isLimitOfReflects s (isLimitOfPreserves (F ⋙ s) hc)⟩⟩⟩⟩


instance [Preregular C] [FinitaryExtensive C]
    [h : ∀ {Y X : C} (f : Y ⟶ X) [EffectiveEpi f], HasPullback f f]
    [PreservesFiniteLimits s] : (coherentTopology C).HasSheafCompose s where
      isSheaf F hF := isSheaf_coherent_of_hasPullbacks_comp (h := h) F s hF


instance [Preregular C] [FinitaryExtensive C] [∀ (X : C), Projective X]
    [PreservesFiniteProducts s] : (coherentTopology C).HasSheafCompose s where
  isSheaf F hF := isSheaf_coherent_of_projective_comp F s hF


