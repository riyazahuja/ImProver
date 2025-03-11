/--
The condition `Precoherent C` is essentially the minimal condition required to define the
coherent coverage on `C`.
-/
class Precoherent : Prop where
  /--
  Given an effective epi family `π₁` over `B₁` and a morphism `f : B₂ ⟶ B₁`, there exists
  an effective epi family `π₂` over `B₂`, such that `π₂` factors through `π₁`.
  -/
  pullback {B₁ B₂ : C} (f : B₂ ⟶ B₁) :
    ∀ (α : Type) [Finite α] (X₁ : α → C) (π₁ : (a : α) → (X₁ a ⟶ B₁)),
      EffectiveEpiFamily X₁ π₁ →
    ∃ (β : Type) (_ : Finite β) (X₂ : β → C) (π₂ : (b : β) → (X₂ b ⟶ B₂)),
      EffectiveEpiFamily X₂ π₂ ∧
      ∃ (i : β → α) (ι : (b :  β) → (X₂ b ⟶ X₁ (i b))),
      ∀ (b : β), ι b ≫ π₁ _ = π₂ _ ≫ f


/--
The coherent coverage on a precoherent category `C`.
-/
def coherentCoverage [Precoherent C] : Coverage C where
  covering B := { S | ∃ (α : Type) (_ : Finite α) (X : α → C) (π : (a : α) → (X a ⟶ B)),
    S = Presieve.ofArrows X π ∧ EffectiveEpiFamily X π }
  pullback := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.641, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Presieve X), Membership …
    -/
    rintro B₁ B₂ f S ⟨α, _, X₁, π₁, rfl, hS⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.641, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      w✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      hS : CategoryTheory.EffectiveEpiFamily X₁ π₁
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun α = …
    -/
    obtain ⟨β,_,X₂,π₂,h,i,ι,hh⟩ := Precoherent.pullback f α X₁ π₁ hS
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.641, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      w✝¹ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      hS : CategoryTheory.EffectiveEpiFamily X₁ π₁
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) B₂
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      i : β → α
      ι : (b : β) → Quiver.Hom (X₂ b) (X₁ (i b))
      hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (π₁ (i b))) (Cate …
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun α = …
    -/
    refine ⟨Presieve.ofArrows X₂ π₂, ⟨β, inferInstance, X₂, π₂, rfl, h⟩, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.641, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      w✝¹ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      hS : CategoryTheory.EffectiveEpiFamily X₁ π₁
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) B₂
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      i : β → α
      ι : (b : β) → Quiver.Hom (X₂ b) (X₁ (i b))
      hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (π₁ (i b))) (Cate …
      ⊢ (CategoryTheory.Presieve.ofArrows X₂ π₂).FactorsThruAlong (CategoryTheory.Pr …
    -/
    rintro _ _ ⟨b⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.641, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      w✝¹ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      hS : CategoryTheory.EffectiveEpiFamily X₁ π₁
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) B₂
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      i : β → α
      ι : (b : β) → Quiver.Hom (X₂ b) (X₁ (i b))
      hh : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) (π₁ (i b))) (Cate …
      Y : C
      b : β
      ⊢ Exists fun W => Exists fun i => Exists fun e => And (CategoryTheory.Presieve …
    -/
    exact ⟨(X₁ (i b)), ι _, π₁ _, ⟨_⟩, hh _⟩
    /-
      🎉 no goals
    -/


/--
The coherent Grothendieck topology on a precoherent category `C`.
-/
def coherentTopology [Precoherent C] : GrothendieckTopology C :=
  Coverage.toGrothendieck _ <| coherentCoverage C


/--
The condition `Preregular C` is property that effective epis can be "pulled back" along any
morphism. This is satisfied e.g. by categories that have pullbacks that preserve effective
epimorphisms (like `Profinite` and `CompHaus`), and categories where every object is projective
(like  `Stonean`).
-/
class Preregular : Prop where
  /--
  For `X`, `Y`, `Z`, `f`, `g` like in the diagram, where `g` is an effective epi, there exists
  an object `W`, an effective epi `h : W ⟶ X` and a morphism `i : W ⟶ Z` making the diagram
  commute.
  ```
  W --i-→ Z
  |       |
  h       g
  ↓       ↓
  X --f-→ Y
  ```
  -/
  exists_fac : ∀ {X Y Z : C} (f : X ⟶ Y) (g : Z ⟶ Y) [EffectiveEpi g],
    (∃ (W : C) (h : W ⟶ X) (_ : EffectiveEpi h) (i : W ⟶ Z), i ≫ g = h ≫ f)


/--
The regular coverage on a regular category `C`.
-/
def regularCoverage [Preregular C] : Coverage C where
  covering B := { S | ∃ (X : C) (f : X ⟶ B), S = Presieve.ofArrows (fun (_ : Unit) ↦ X)
    (fun (_ : Unit) ↦ f) ∧ EffectiveEpi f }
  pullback := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
      inst✝ : CategoryTheory.Preregular C
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Presieve X), Membership …
    -/
    intro X Y f S ⟨Z, π, hπ, h_epi⟩
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
      inst✝ : CategoryTheory.Preregular C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      Z : C
      π : Quiver.Hom Z X
      hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
      h_epi : CategoryTheory.EffectiveEpi π
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun X = …
    -/
    have := Preregular.exists_fac f π
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
      inst✝ : CategoryTheory.Preregular C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      Z : C
      π : Quiver.Hom Z X
      hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
      h_epi : CategoryTheory.EffectiveEpi π
      this : Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Cat …
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun X = …
    -/
    obtain ⟨W, h, _, i, this⟩ := this
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
      inst✝ : CategoryTheory.Preregular C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      Z : C
      π : Quiver.Hom Z X
      hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
      h_epi : CategoryTheory.EffectiveEpi π
      W : C
      h : Quiver.Hom W Y
      w✝ : CategoryTheory.EffectiveEpi h
      i : Quiver.Hom W Z
      this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun X = …
    -/
    refine ⟨Presieve.singleton h, ⟨?_, ?_⟩⟩
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        Z : C
        π : Quiver.Hom Z X
        hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ Membership.mem ((fun B => setOf fun S => Exists fun X => Exists fun f => And …
      -/
    · exact ⟨W, h, by {rw [Presieve.ofArrows_pUnit h]}, inferInstance⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        Z : C
        π : Quiver.Hom Z X
        hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ (CategoryTheory.Presieve.singleton h).FactorsThruAlong S f
      -/
    · intro W g hg
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        Z : C
        π : Quiver.Hom Z X
        hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
        h_epi : CategoryTheory.EffectiveEpi π
        W✝ : C
        h : Quiver.Hom W✝ Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W✝ Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        W : C
        g : Quiver.Hom W Y
        hg : CategoryTheory.Presieve.singleton h g
        ⊢ Exists fun W_1 => Exists fun i => Exists fun e => And (S e) (Eq (CategoryThe …
      -/
      cases hg
      /-
        case intro.intro.intro.intro.refine_2.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        Z : C
        π : Quiver.Hom Z X
        hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ Exists fun W_1 => Exists fun i => Exists fun e => And (S e) (Eq (CategoryThe …
      -/
      refine ⟨Z, i, π, ⟨?_, this⟩⟩
      /-
        case intro.intro.intro.intro.refine_2.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        Z : C
        π : Quiver.Hom Z X
        hπ : Eq S (CategoryTheory.Presieve.ofArrows (fun x => Z) fun x => π)
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ S π
      -/
      cases hπ
      /-
        case intro.intro.intro.intro.refine_2.mk.refl
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        Z : C
        π : Quiver.Hom Z X
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ CategoryTheory.Presieve.ofArrows (fun x => Z) (fun x => π) π
      -/
      rw [Presieve.ofArrows_pUnit]
      /-
        case intro.intro.intro.intro.refine_2.mk.refl
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.2656, u_1} C
        inst✝ : CategoryTheory.Preregular C
        X Y : C
        f : Quiver.Hom Y X
        Z : C
        π : Quiver.Hom Z X
        h_epi : CategoryTheory.EffectiveEpi π
        W : C
        h : Quiver.Hom W Y
        w✝ : CategoryTheory.EffectiveEpi h
        i : Quiver.Hom W Z
        this : Eq (CategoryTheory.CategoryStruct.comp i π) (CategoryTheory.CategoryStr …
        ⊢ CategoryTheory.Presieve.singleton π π
      -/
      exact Presieve.singleton.mk
      /-
        🎉 no goals
      -/


/--
The regular Grothendieck topology on a preregular category `C`.
-/
def regularTopology [Preregular C] : GrothendieckTopology C :=
  Coverage.toGrothendieck _ <| regularCoverage C


/--
The extensive coverage on an extensive category `C`

TODO: use general colimit API instead of `IsIso (Sigma.desc π)`
-/
def extensiveCoverage [FinitaryPreExtensive C] : Coverage C where
  covering B := { S | ∃ (α : Type) (_ : Finite α) (X : α → C) (π : (a : α) → (X a ⟶ B)),
    S = Presieve.ofArrows X π ∧ IsIso (Sigma.desc π) }
  pullback := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Presieve X), Membership …
    -/
    intro X Y f S ⟨α, hα, Z, π, hS, h_iso⟩
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      α : Type
      hα : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
      h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun α = …
    -/
    let Z' : α → C := fun a ↦ pullback f (π a)
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      α : Type
      hα : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
      h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun α = …
    -/
    let π' : (a : α) → Z' a ⟶ Y := fun a ↦ pullback.fst _ _
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
      inst✝ : CategoryTheory.FinitaryPreExtensive C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      α : Type
      hα : Finite α
      Z : α → C
      π : (a : α) → Quiver.Hom (Z a) X
      hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
      h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
      π' : (a : α) → Quiver.Hom (Z' a) Y := fun a => CategoryTheory.Limits.pullback. …
      ⊢ Exists fun T => And (Membership.mem ((fun B => setOf fun S => Exists fun α = …
    -/
    refine ⟨@Presieve.ofArrows C _ _ α Z' π', ⟨?_, ?_⟩⟩
      /-
        case refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y := fun a => CategoryTheory.Limits.pullback. …
        ⊢ Membership.mem ((fun B => setOf fun S => Exists fun α => Exists fun x => Exi …
      -/
    · constructor
      /-
        case refine_1.h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y := fun a => CategoryTheory.Limits.pullback. …
        ⊢ Exists fun x => Exists fun X => Exists fun π => And (Eq (CategoryTheory.Pres …
      -/
      exact ⟨hα, Z', π', ⟨by simp only, FinitaryPreExtensive.sigma_desc_iso (fun x => π x) f h_iso⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y := fun a => CategoryTheory.Limits.pullback. …
        ⊢ (CategoryTheory.Presieve.ofArrows Z' π').FactorsThruAlong S f
      -/
    · intro W g hg
      /-
        case refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y : C
        f : Quiver.Hom Y X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y := fun a => CategoryTheory.Limits.pullback. …
        W : C
        g : Quiver.Hom W Y
        hg : CategoryTheory.Presieve.ofArrows Z' π' g
        ⊢ Exists fun W_1 => Exists fun i => Exists fun e => And (S e) (Eq (CategoryThe …
      -/
      rcases hg with ⟨a⟩
      /-
        case refine_2.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y✝ : C
        f : Quiver.Hom Y✝ X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y✝ := fun a => CategoryTheory.Limits.pullback …
        Y : C
        a : α
        ⊢ Exists fun W => Exists fun i => Exists fun e => And (S e) (Eq (CategoryTheor …
      -/
      refine ⟨Z a, pullback.snd _ _, π a, ?_, by rw [CategoryTheory.Limits.pullback.condition]⟩
      /-
        case refine_2.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y✝ : C
        f : Quiver.Hom Y✝ X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y✝ := fun a => CategoryTheory.Limits.pullback …
        Y : C
        a : α
        ⊢ S (π a)
      -/
      rw [hS]
      /-
        case refine_2.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.4415, u_1} C
        inst✝ : CategoryTheory.FinitaryPreExtensive C
        X Y✝ : C
        f : Quiver.Hom Y✝ X
        S : CategoryTheory.Presieve X
        α : Type
        hα : Finite α
        Z : α → C
        π : (a : α) → Quiver.Hom (Z a) X
        hS : Eq S (CategoryTheory.Presieve.ofArrows Z π)
        h_iso : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
        Z' : α → C := fun a => CategoryTheory.Limits.pullback f (π a)
        π' : (a : α) → Quiver.Hom (Z' a) Y✝ := fun a => CategoryTheory.Limits.pullback …
        Y : C
        a : α
        ⊢ CategoryTheory.Presieve.ofArrows Z π (π a)
      -/
      exact Presieve.ofArrows.mk a
      /-
        🎉 no goals
      -/


/--
The extensive Grothendieck topology on a finitary pre-extensive category `C`.
-/
def extensiveTopology [FinitaryPreExtensive C] : GrothendieckTopology C :=
  Coverage.toGrothendieck _ <| extensiveCoverage C


