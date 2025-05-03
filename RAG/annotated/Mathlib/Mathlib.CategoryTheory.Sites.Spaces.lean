/-- The Grothendieck topology associated to a topological space. -/
def grothendieckTopology : GrothendieckTopology (Opens T) where
  sieves X S := ∀ x ∈ X, ∃ (U : _) (f : U ⟶ X), S f ∧ x ∈ U
  top_mem' _ _ hx := ⟨_, 𝟙 _, trivial, hx⟩
  pullback_stable' X Y S f hf y hy := by
    /-
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hf : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      y : T
      hy : Membership.mem Y y
      ⊢ Exists fun U => Exists fun f_1 => And ((CategoryTheory.Sieve.pullback f S).a …
    -/
    rcases hf y (f.le hy) with ⟨U, g, hg, hU⟩
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hf : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      y : T
      hy : Membership.mem Y y
      U : TopologicalSpace.Opens T
      g : Quiver.Hom U X
      hg : S.arrows g
      hU : Membership.mem U y
      ⊢ Exists fun U => Exists fun f_1 => And ((CategoryTheory.Sieve.pullback f S).a …
    -/
    refine ⟨U ⊓ Y, homOfLE inf_le_right, ?_, hU, hy⟩
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hf : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      y : T
      hy : Membership.mem Y y
      U : TopologicalSpace.Opens T
      g : Quiver.Hom U X
      hg : S.arrows g
      hU : Membership.mem U y
      ⊢ (CategoryTheory.Sieve.pullback f S).arrows (CategoryTheory.homOfLE ⋯)
    -/
    apply S.downward_closed hg (homOfLE inf_le_left)
    /-
      🎉 no goals
    -/
  transitive' X S hS R hR x hx := by
    /-
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : TopologicalSpace.Opens T⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membe …
      x : T
      hx : Membership.mem X x
      ⊢ Exists fun U => Exists fun f => And (R.arrows f) (Membership.mem U x)
    -/
    rcases hS x hx with ⟨U, f, hf, hU⟩
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : TopologicalSpace.Opens T⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membe …
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hf : S.arrows f
      hU : Membership.mem U x
      ⊢ Exists fun U => Exists fun f => And (R.arrows f) (Membership.mem U x)
    -/
    rcases hR hf _ hU with ⟨V, g, hg, hV⟩
    /-
      case intro.intro.intro.intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun X S => ∀ (x : T), Membership.mem X x → Exists fun U  …
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : TopologicalSpace.Opens T⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membe …
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hf : S.arrows f
      hU : Membership.mem U x
      V : TopologicalSpace.Opens T
      g : Quiver.Hom V U
      hg : (CategoryTheory.Sieve.pullback f R).arrows g
      hV : Membership.mem V x
      ⊢ Exists fun U => Exists fun f => And (R.arrows f) (Membership.mem U x)
    -/
    exact ⟨_, g ≫ f, hg, hV⟩
    /-
      🎉 no goals
    -/


/-- The Grothendieck pretopology associated to a topological space. -/
def pretopology : Pretopology (Opens T) where
  coverings X R := ∀ x ∈ X, ∃ (U : _) (f : U ⟶ X), R f ∧ x ∈ U
  has_isos _ _ f _ _ hx := ⟨_, _, Presieve.singleton_self _, (inv f).le hx⟩
  pullbacks X Y f S hS x hx := by
    /-
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      x : T
      hx : Membership.mem Y x
      ⊢ Exists fun U => Exists fun f_1 => And (CategoryTheory.Presieve.pullbackArrow …
    -/
    rcases hS _ (f.le hx) with ⟨U, g, hg, hU⟩
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      x : T
      hx : Membership.mem Y x
      U : TopologicalSpace.Opens T
      g : Quiver.Hom U X
      hg : S g
      hU : Membership.mem U x
      ⊢ Exists fun U => Exists fun f_1 => And (CategoryTheory.Presieve.pullbackArrow …
    -/
    refine ⟨_, _, Presieve.pullbackArrows.mk _ _ hg, ?_⟩
    have : U ⊓ Y ≤ pullback g f :=
      leOfHom (pullback.lift (homOfLE inf_le_left) (homOfLE inf_le_right) rfl)
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X Y : TopologicalSpace.Opens T
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      x : T
      hx : Membership.mem Y x
      U : TopologicalSpace.Opens T
      g : Quiver.Hom U X
      hg : S g
      hU : Membership.mem U x
      this : LE.le (Min.min U Y) (CategoryTheory.Limits.pullback g f)
      ⊢ Membership.mem (CategoryTheory.Limits.pullback g f) x
    -/
    apply this ⟨hU, hx⟩
    /-
      🎉 no goals
    -/
  transitive X S Ti hS hTi x hx := by
    /-
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : TopologicalSpace.Opens T⦄ → (f : Quiver.Hom Y X) → S f → CategoryThe …
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      hTi : ∀ ⦃Y : TopologicalSpace.Opens T⦄ (f : Quiver.Hom Y X) (H : S f), Members …
      x : T
      hx : Membership.mem X x
      ⊢ Exists fun U => Exists fun f => And (S.bind Ti f) (Membership.mem U x)
    -/
    rcases hS x hx with ⟨U, f, hf, hU⟩
    /-
      case intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : TopologicalSpace.Opens T⦄ → (f : Quiver.Hom Y X) → S f → CategoryThe …
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      hTi : ∀ ⦃Y : TopologicalSpace.Opens T⦄ (f : Quiver.Hom Y X) (H : S f), Members …
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hf : S f
      hU : Membership.mem U x
      ⊢ Exists fun U => Exists fun f => And (S.bind Ti f) (Membership.mem U x)
    -/
    rcases hTi f hf x hU with ⟨V, g, hg, hV⟩
    /-
      case intro.intro.intro.intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      S : CategoryTheory.Presieve X
      Ti : ⦃Y : TopologicalSpace.Opens T⦄ → (f : Quiver.Hom Y X) → S f → CategoryThe …
      hS : Membership.mem ((fun X R => ∀ (x : T), Membership.mem X x → Exists fun U  …
      hTi : ∀ ⦃Y : TopologicalSpace.Opens T⦄ (f : Quiver.Hom Y X) (H : S f), Members …
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hf : S f
      hU : Membership.mem U x
      V : TopologicalSpace.Opens T
      g : Quiver.Hom V U
      hg : Ti f hf g
      hV : Membership.mem V x
      ⊢ Exists fun U => Exists fun f => And (S.bind Ti f) (Membership.mem U x)
    -/
    exact ⟨_, _, ⟨_, g, f, hf, hg, rfl⟩, hV⟩
    /-
      🎉 no goals
    -/


/-- The pretopology associated to a space is the largest pretopology that
    generates the Grothendieck topology associated to the space. -/
@[simp]
theorem pretopology_ofGrothendieck :
    Pretopology.ofGrothendieck _ (Opens.grothendieckTopology T) = Opens.pretopology T := by
  /-
    T : Type u
    inst✝ : TopologicalSpace T
    ⊢ Eq (CategoryTheory.Pretopology.ofGrothendieck (TopologicalSpace.Opens T) (Op …
  -/
  apply le_antisymm
    /-
      case a
      T : Type u
      inst✝ : TopologicalSpace T
      ⊢ LE.le (CategoryTheory.Pretopology.ofGrothendieck (TopologicalSpace.Opens T)  …
    -/
  · intro X R hR x hx
    /-
      case a
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((CategoryTheory.Pretopology.ofGrothendieck (TopologicalSp …
      x : T
      hx : Membership.mem X x
      ⊢ Exists fun U => Exists fun f => And (R f) (Membership.mem U x)
    -/
    rcases hR x hx with ⟨U, f, ⟨V, g₁, g₂, hg₂, _⟩, hU⟩
    /-
      case a.intro.intro.intro.intro.intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((CategoryTheory.Pretopology.ofGrothendieck (TopologicalSp …
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hU : Membership.mem U x
      V : TopologicalSpace.Opens T
      g₁ : Quiver.Hom U V
      g₂ : Quiver.Hom V X
      hg₂ : R g₂
      right✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ g₂) f
      ⊢ Exists fun U => Exists fun f => And (R f) (Membership.mem U x)
    -/
    exact ⟨V, g₂, hg₂, g₁.le hU⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      T : Type u
      inst✝ : TopologicalSpace T
      ⊢ LE.le (Opens.pretopology T) (CategoryTheory.Pretopology.ofGrothendieck (Topo …
    -/
  · intro X R hR x hx
    /-
      case a
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((Opens.pretopology T).coverings X) R
      x : T
      hx : Membership.mem X x
      ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.generate R).arrow …
    -/
    rcases hR x hx with ⟨U, f, hf, hU⟩
    /-
      case a.intro.intro.intro
      T : Type u
      inst✝ : TopologicalSpace T
      X : TopologicalSpace.Opens T
      R : CategoryTheory.Presieve X
      hR : Membership.mem ((Opens.pretopology T).coverings X) R
      x : T
      hx : Membership.mem X x
      U : TopologicalSpace.Opens T
      f : Quiver.Hom U X
      hf : R f
      hU : Membership.mem U x
      ⊢ Exists fun U => Exists fun f => And ((CategoryTheory.Sieve.generate R).arrow …
    -/
    exact ⟨U, f, Sieve.le_generate R U hf, hU⟩
    /-
      🎉 no goals
    -/


/-- The pretopology associated to a space induces the Grothendieck topology associated to the space.
-/
@[simp]
theorem pretopology_toGrothendieck :
    Pretopology.toGrothendieck _ (Opens.pretopology T) = Opens.grothendieckTopology T := by
  /-
    T : Type u
    inst✝ : TopologicalSpace T
    ⊢ Eq (CategoryTheory.Pretopology.toGrothendieck (TopologicalSpace.Opens T) (Op …
  -/
  rw [← pretopology_ofGrothendieck]
  /-
    T : Type u
    inst✝ : TopologicalSpace T
    ⊢ Eq (CategoryTheory.Pretopology.toGrothendieck (TopologicalSpace.Opens T) (Ca …
  -/
  apply (Pretopology.gi (Opens T)).l_u_eq
  /-
    🎉 no goals
  -/


