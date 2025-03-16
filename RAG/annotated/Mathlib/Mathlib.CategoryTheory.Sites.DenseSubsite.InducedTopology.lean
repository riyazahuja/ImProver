/-- We say that a functor `C ⥤ D` into a site is "locally dense" if
for each covering sieve `T` in `D`, `T ∩ mor(C)` generates a covering sieve in `D`.
-/
class LocallyCoverDense : Prop where
  functorPushforward_functorPullback_mem :
    ∀ ⦃X : C⦄ (T : K (G.obj X)), (T.val.functorPullback G).functorPushforward G ∈ K (G.obj X)


theorem pushforward_cover_iff_cover_pullback [G.Full] [G.Faithful] {X : C} (S : Sieve X) :
    K _ (S.functorPushforward G) ↔ ∃ T : K (G.obj X), T.val.functorPullback G = S := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    G : CategoryTheory.Functor C D
    K : CategoryTheory.GrothendieckTopology D
    inst✝² : G.LocallyCoverDense K
    inst✝¹ : G.Full
    inst✝ : G.Faithful
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (K (G.obj X) (CategoryTheory.Sieve.functorPushforward G S)) (Exists fun  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.Full
      inst✝ : G.Faithful
      X : C
      S : CategoryTheory.Sieve X
      ⊢ K (G.obj X) (CategoryTheory.Sieve.functorPushforward G S) → Exists fun T =>  …
    -/
  · intro hS
    /-
      case mp
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.Full
      inst✝ : G.Faithful
      X : C
      S : CategoryTheory.Sieve X
      hS : K (G.obj X) (CategoryTheory.Sieve.functorPushforward G S)
      ⊢ Exists fun T => Eq (CategoryTheory.Sieve.functorPullback G ↑T) S
    -/
    exact ⟨⟨_, hS⟩, (Sieve.fullyFaithfulFunctorGaloisCoinsertion G X).u_l_eq S⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.Full
      inst✝ : G.Faithful
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (Exists fun T => Eq (CategoryTheory.Sieve.functorPullback G ↑T) S) → K (G.ob …
    -/
  · rintro ⟨T, rfl⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_4, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.Full
      inst✝ : G.Faithful
      X : C
      T : ↑(K (G.obj X))
      ⊢ K (G.obj X) (CategoryTheory.Sieve.functorPushforward G (CategoryTheory.Sieve …
    -/
    exact LocallyCoverDense.functorPushforward_functorPullback_mem T
    /-
      🎉 no goals
    -/


/-- If a functor `G : C ⥤ (D, K)` is fully faithful and locally dense,
then the set `{ T ∩ mor(C) | T ∈ K }` is a grothendieck topology of `C`.
-/
@[simps]
def inducedTopology : GrothendieckTopology C where
  sieves _ S := K _ (S.functorPushforward G)
  top_mem' X := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      ⊢ Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPushfor …
    -/
    change K _ _
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      ⊢ K (G.obj X) (CategoryTheory.Sieve.functorPushforward G Top.top)
    -/
    rw [Sieve.functorPushforward_top]
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      ⊢ K (G.obj X) Top.top
    -/
    exact K.top_mem _
    /-
      🎉 no goals
    -/
  pullback_stable' X Y S iYX hS := by
    apply K.transitive (LocallyCoverDense.functorPushforward_functorPullback_mem
      ⟨_, K.pullback_stable (G.map iYX) hS⟩)
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      ⊢ ∀ ⦃Y_1 : D⦄ ⦃f : Quiver.Hom Y_1 (G.obj Y)⦄, (CategoryTheory.Sieve.functorPus …
    -/
    rintro Z _ ⟨U, iUY, iZU, ⟨W, iWX, iUW, hiWX, e₁⟩, rfl⟩
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      Z : D
      U : C
      iUY : Quiver.Hom U Y
      iZU : Quiver.Hom Z (G.obj U)
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      ⊢ Membership.mem (K Z) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
    -/
    rw [Sieve.pullback_comp]
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      Z : D
      U : C
      iUY : Quiver.Hom U Y
      iZU : Quiver.Hom Z (G.obj U)
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      ⊢ Membership.mem (K Z) (CategoryTheory.Sieve.pullback iZU (CategoryTheory.Siev …
    -/
    apply K.pullback_stable
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      Z : D
      U : C
      iUY : Quiver.Hom U Y
      iZU : Quiver.Hom Z (G.obj U)
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      ⊢ Membership.mem (K (G.obj U)) (CategoryTheory.Sieve.pullback (G.map iUY) (Cat …
    -/
    clear iZU Z
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      ⊢ Membership.mem (K (G.obj U)) (CategoryTheory.Sieve.pullback (G.map iUY) (Cat …
    -/
    apply K.transitive (G.functorPushforward_imageSieve_mem _ iUW)
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      ⊢ ∀ ⦃Y_1 : D⦄ ⦃f : Quiver.Hom Y_1 (G.obj U)⦄, (CategoryTheory.Sieve.functorPus …
    -/
    rintro Z _ ⟨U₁, iU₁U, iZU₁, ⟨iU₁W, e₂⟩, rfl⟩
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS.h.intro.intro.intro. …
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      Z : D
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iZU₁ : Quiver.Hom Z (G.obj U₁)
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      ⊢ Membership.mem (K Z) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
    -/
    rw [Sieve.pullback_comp]
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS.h.intro.intro.intro. …
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      Z : D
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iZU₁ : Quiver.Hom Z (G.obj U₁)
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      ⊢ Membership.mem (K Z) (CategoryTheory.Sieve.pullback iZU₁ (CategoryTheory.Sie …
    -/
    apply K.pullback_stable
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro.hS.h.intro.intro.intro. …
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      Z : D
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iZU₁ : Quiver.Hom Z (G.obj U₁)
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      ⊢ Membership.mem (K (G.obj U₁)) (CategoryTheory.Sieve.pullback (G.map iU₁U) (C …
    -/
    clear iZU₁ Z
    apply K.superset_covering ?_ (G.functorPushforward_equalizer_mem _
      (iU₁U ≫ iUY ≫ iYX) (iU₁W ≫ iWX) (by simp [e₁, e₂]))
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      ⊢ LE.le (CategoryTheory.Sieve.functorPushforward G (CategoryTheory.Sieve.equal …
    -/
    rintro Z _ ⟨U₂, iU₂U₁, iZU₂, e₃ : _ = _, rfl⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      Z : D
      U₂ : C
      iU₂U₁ : Quiver.Hom U₂ U₁
      iZU₂ : Quiver.Hom Z (G.obj U₂)
      e₃ : Eq (CategoryTheory.CategoryStruct.comp iU₂U₁ (CategoryTheory.CategoryStru …
      ⊢ (CategoryTheory.Sieve.pullback (G.map iU₁U) (CategoryTheory.Sieve.pullback ( …
    -/
    refine ⟨_, iU₂U₁ ≫ iU₁U ≫ iUY, iZU₂, ?_, by simp⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X Y : C
      S : CategoryTheory.Sieve X
      iYX : Quiver.Hom Y X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      U : C
      iUY : Quiver.Hom U Y
      W : C
      iWX : Quiver.Hom W X
      iUW : Quiver.Hom (G.obj U) (G.obj W)
      hiWX : S.arrows iWX
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (G.map iUY) (G.map iYX)) (Category …
      U₁ : C
      iU₁U : Quiver.Hom U₁ U
      iU₁W : Quiver.Hom U₁ W
      e₂ : Eq (G.map iU₁W) (CategoryTheory.CategoryStruct.comp (G.map iU₁U) iUW)
      Z : D
      U₂ : C
      iU₂U₁ : Quiver.Hom U₂ U₁
      iZU₂ : Quiver.Hom Z (G.obj U₂)
      e₃ : Eq (CategoryTheory.CategoryStruct.comp iU₂U₁ (CategoryTheory.CategoryStru …
      ⊢ (CategoryTheory.Sieve.pullback iYX S).arrows (CategoryTheory.CategoryStruct. …
    -/
    simpa [e₃] using S.downward_closed hiWX (iU₂U₁ ≫ iU₁W)
    /-
      🎉 no goals
    -/
  transitive' X S hS S' H' := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      ⊢ Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPushfor …
    -/
    apply K.transitive hS
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      ⊢ ∀ ⦃Y : D⦄ ⦃f : Quiver.Hom Y (G.obj X)⦄, (CategoryTheory.Sieve.functorPushfor …
    -/
    rintro Y _ ⟨Z, g, i, hg, rfl⟩
    /-
      case h.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg : S.arrows g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
    -/
    rw [Sieve.pullback_comp]
    /-
      case h.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg : S.arrows g
      ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback i (CategoryTheory.Sieve. …
    -/
    apply K.pullback_stable i
    /-
      case h.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg : S.arrows g
      ⊢ Membership.mem (K (G.obj Z)) (CategoryTheory.Sieve.pullback (G.map g) (Categ …
    -/
    refine K.superset_covering ?_ (H' hg)
    /-
      case h.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg : S.arrows g
      ⊢ LE.le (CategoryTheory.Sieve.functorPushforward G (CategoryTheory.Sieve.pullb …
    -/
    rintro W _ ⟨Z', g', i', hg, rfl⟩
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg✝ : S.arrows g
      W : D
      Z' : C
      g' : Quiver.Hom Z' Z
      i' : Quiver.Hom W (G.obj Z')
      hg : (CategoryTheory.Sieve.pullback g S').arrows g'
      ⊢ (CategoryTheory.Sieve.pullback (G.map g) (CategoryTheory.Sieve.functorPushfo …
    -/
    refine ⟨Z', g' ≫ g , i', hg, ?_⟩
    /-
      case h.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.2086, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.2093, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun x S => K (G.obj x) (CategoryTheory.Sieve.functorPush …
      S' : CategoryTheory.Sieve X
      H' : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun x S =>  …
      Y : D
      Z : C
      g : Quiver.Hom Z X
      i : Quiver.Hom Y (G.obj Z)
      hg✝ : S.arrows g
      W : D
      Z' : C
      g' : Quiver.Hom Z' Z
      i' : Quiver.Hom W (G.obj Z')
      hg : (CategoryTheory.Sieve.pullback g S').arrows g'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
lemma mem_inducedTopology_sieves_iff {X : C} (S : Sieve X) :
    S ∈ (G.inducedTopology K) X ↔ (S.functorPushforward G) ∈ K (G.obj X) :=
  Iff.rfl


/-- `G` is cover-lifting wrt the induced topology. -/
instance inducedTopology_isCocontinuous : G.IsCocontinuous (G.inducedTopology K) K :=
  ⟨@fun _ S hS => LocallyCoverDense.functorPushforward_functorPullback_mem ⟨S, hS⟩⟩


/-- `G` is cover-preserving wrt the induced topology. -/
theorem inducedTopology_coverPreserving : CoverPreserving (G.inducedTopology K) K G :=
  ⟨@fun _ _ hS => hS⟩


instance (priority := 900) locallyCoverDense_of_isCoverDense [G.IsCoverDense K] :
    G.LocallyCoverDense K where
  functorPushforward_functorPullback_mem _ _ :=
    IsCoverDense.functorPullback_pushforward_covering _


instance (priority := 900) [G.IsCoverDense K] : G.IsDenseSubsite (G.inducedTopology K) K where
  functorPushforward_mem_iff := Iff.rfl


@[deprecated (since := "2024-07-23")]
alias inducedTopologyOfIsCoverDense := inducedTopology


instance over_forget_locallyCoverDense (X : C) : (Over.forget X).LocallyCoverDense J where
  functorPushforward_functorPullback_mem Y T := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      Y : CategoryTheory.Over X
      T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
      ⊢ Membership.mem (J ((CategoryTheory.Over.forget X).obj Y)) (CategoryTheory.Si …
    -/
    convert T.property
    /-
      case h.e'_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      Y : CategoryTheory.Over X
      T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
      ⊢ Eq (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.forget X) ( …
    -/
    ext Z f
    /-
      case h.e'_5.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type v
      inst✝³ : CategoryTheory.Category.{u, v} A
      inst✝² : G.LocallyCoverDense K
      inst✝¹ : G.IsLocallyFull K
      inst✝ : G.IsLocallyFaithful K
      X : C
      Y : CategoryTheory.Over X
      T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
      Z : C
      f : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
      ⊢ Iff ((CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.forget X) …
    -/
    constructor
      /-
        case h.e'_5.h.mp
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type v
        inst✝³ : CategoryTheory.Category.{u, v} A
        inst✝² : G.LocallyCoverDense K
        inst✝¹ : G.IsLocallyFull K
        inst✝ : G.IsLocallyFaithful K
        X : C
        Y : CategoryTheory.Over X
        T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
        Z : C
        f : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        ⊢ (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.forget X) (Cat …
      -/
    · rintro ⟨_, _, g', hg, rfl⟩
      /-
        case h.e'_5.h.mp.intro.intro.intro.intro
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type v
        inst✝³ : CategoryTheory.Category.{u, v} A
        inst✝² : G.LocallyCoverDense K
        inst✝¹ : G.IsLocallyFull K
        inst✝ : G.IsLocallyFaithful K
        X : C
        Y : CategoryTheory.Over X
        T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
        Z : C
        w✝¹ : CategoryTheory.Over X
        w✝ : Quiver.Hom w✝¹ Y
        g' : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj w✝¹)
        hg : (CategoryTheory.Sieve.functorPullback (CategoryTheory.Over.forget X) ↑T). …
        ⊢ (↑T).arrows (CategoryTheory.CategoryStruct.comp g' ((CategoryTheory.Over.for …
      -/
      exact T.val.downward_closed hg g'
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.mpr
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type v
        inst✝³ : CategoryTheory.Category.{u, v} A
        inst✝² : G.LocallyCoverDense K
        inst✝¹ : G.IsLocallyFull K
        inst✝ : G.IsLocallyFaithful K
        X : C
        Y : CategoryTheory.Over X
        T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
        Z : C
        f : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        ⊢ (↑T).arrows f → (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Ove …
      -/
    · intro hf
      /-
        case h.e'_5.h.mpr
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.13041, u_2} D
        G : CategoryTheory.Functor C D
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        A : Type v
        inst✝³ : CategoryTheory.Category.{u, v} A
        inst✝² : G.LocallyCoverDense K
        inst✝¹ : G.IsLocallyFull K
        inst✝ : G.IsLocallyFaithful K
        X : C
        Y : CategoryTheory.Over X
        T : ↑(J ((CategoryTheory.Over.forget X).obj Y))
        Z : C
        f : Quiver.Hom Z ((CategoryTheory.Over.forget X).obj Y)
        hf : (↑T).arrows f
        ⊢ (CategoryTheory.Sieve.functorPushforward (CategoryTheory.Over.forget X) (Cat …
      -/
      exact ⟨Over.mk (f ≫ Y.hom), Over.homMk f, 𝟙 _, hf, (Category.id_comp _).symm⟩
      /-
        🎉 no goals
      -/


/-- Cover-dense functors induces an equivalence of categories of sheaves.

This is known as the comparison lemma. It requires that the sites are small and the value category
is complete.
-/
noncomputable def sheafInducedTopologyEquivOfIsCoverDense
    [G.IsCoverDense K] [∀ (X : Dᵒᵖ), HasLimitsOfShape (StructuredArrow X G.op) A] :
    Sheaf (G.inducedTopology K) A ≌ Sheaf K A :=
  Functor.IsDenseSubsite.sheafEquiv G
    (G.inducedTopology K) K A


