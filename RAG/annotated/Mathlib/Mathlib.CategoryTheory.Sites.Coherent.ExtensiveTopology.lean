lemma extensiveTopology.mem_sieves_iff_contains_colimit_cofan {X : C} (S : Sieve X) :
    S ∈ (extensiveTopology C) X ↔
      (∃ (α : Type) (_ : Finite α) (Y : α → C) (π : (a : α) → (Y a ⟶ X)),
        Nonempty (IsColimit (Cofan.mk X π)) ∧ (∀ a : α, (S.arrows) (π a))) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.extensiveTopology C) X) S) (Exists fun  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.extensiveTopology C) X) S → Exists fun α =>  …
    -/
  · intro h
    induction h with
    | of X S hS =>
      obtain ⟨α, _, Y, π, h, h'⟩ := hS
      refine ⟨α, inferInstance, Y, π, ?_, fun a ↦ ?_⟩
      · have : IsIso (Sigma.desc (Cofan.mk X π).inj) := by simpa using h'
        exact ⟨Cofan.isColimitOfIsIsoSigmaDesc (Cofan.mk X π)⟩
      · obtain ⟨rfl, _⟩ := h
        exact ⟨Y a, 𝟙 Y a, π a, Presieve.ofArrows.mk a, by simp⟩
    | top X =>
      refine ⟨Unit, inferInstance, fun _ => X, fun _ => (𝟙 X), ⟨?_⟩, by simp⟩
      have : IsIso (Sigma.desc (Cofan.mk X fun (_ : Unit) ↦ 𝟙 X).inj) := by
        have : IsIso (coproductUniqueIso (fun () => X)).hom := inferInstance
        exact this
      exact Cofan.isColimitOfIsIsoSigmaDesc (Cofan.mk X _)
    | transitive X R S _ _ a b =>
      obtain ⟨α, w, Y₁, π, h, h'⟩ := a
      choose β _ Y_n π_n H using fun a => b (h' a)
      exact ⟨(Σ a, β a), inferInstance, fun ⟨a,b⟩ => Y_n a b, fun ⟨a, b⟩ => (π_n a b) ≫ (π a),
        ⟨Limits.Cofan.isColimitTrans _ h.some _ (fun a ↦ (H a).1.some)⟩,
        fun c => (H c.fst).2 c.snd⟩
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (Exists fun α => Exists fun x => Exists fun Y => Exists fun π => And (Nonemp …
    -/
  · intro ⟨α, _, Y, π, h, h'⟩
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X : C
      S : CategoryTheory.Sieve X
      α : Type
      w✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
      h' : ∀ (a : α), S.arrows (π a)
      ⊢ Membership.mem ((CategoryTheory.extensiveTopology C) X) S
    -/
    apply (extensiveCoverage C).mem_toGrothendieck_sieves_of_superset (R := Presieve.ofArrows Y π)
      /-
        case mpr.h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X : C
        S : CategoryTheory.Sieve X
        α : Type
        w✝ : Finite α
        Y : α → C
        π : (a : α) → Quiver.Hom (Y a) X
        h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
        h' : ∀ (a : α), S.arrows (π a)
        ⊢ LE.le (CategoryTheory.Presieve.ofArrows Y π) S.arrows
      -/
    · exact fun _ _ hh ↦ by cases hh; exact h' _
      /-
        🎉 no goals
      -/
      /-
        case mpr.hR
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X : C
        S : CategoryTheory.Sieve X
        α : Type
        w✝ : Finite α
        Y : α → C
        π : (a : α) → Quiver.Hom (Y a) X
        h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
        h' : ∀ (a : α), S.arrows (π a)
        ⊢ Membership.mem ((CategoryTheory.extensiveCoverage C).covering X) (CategoryTh …
      -/
    · refine ⟨α, inferInstance, Y, π, rfl, ?_⟩
      rw [show IsIso (Sigma.desc π) ↔ _ from
        Limits.Cofan.isColimit_iff_isIso_sigmaDesc (c := Cofan.mk X π)]
      /-
        case mpr.hR
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X : C
        S : CategoryTheory.Sieve X
        α : Type
        w✝ : Finite α
        Y : α → C
        π : (a : α) → Quiver.Hom (Y a) X
        h : Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk  …
        h' : ∀ (a : α), S.arrows (π a)
        ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X  …
      -/
      exact h
      /-
        🎉 no goals
      -/


