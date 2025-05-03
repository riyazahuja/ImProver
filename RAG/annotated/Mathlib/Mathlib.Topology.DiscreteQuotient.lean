/-- The type of discrete quotients of a topological space. -/
@[ext] -- Porting note: in Lean 4, uses projection to `r` instead of `Setoid`.
structure DiscreteQuotient (X : Type*) [TopologicalSpace X] extends Setoid X where
  /-- For every point `x`, the set `{ y | Rel x y }` is an open set. -/
  protected isOpen_setOf_rel : ∀ x, IsOpen (setOf (toSetoid x))


lemma toSetoid_injective : Function.Injective (@toSetoid X _)
                            /-
                              X : Type u_2
                              inst✝ : TopologicalSpace X
                              toSetoid✝¹ : Setoid X
                              isOpen_setOf_rel✝¹ : ∀ (x : X), IsOpen (setOf (toSetoid✝¹ x))
                              toSetoid✝ : Setoid X
                              isOpen_setOf_rel✝ : ∀ (x : X), IsOpen (setOf (toSetoid✝ x))
                              x✝ : Eq { toSetoid := toSetoid✝¹, isOpen_setOf_rel := isOpen_setOf_rel✝¹ }.toS …
                              ⊢ Eq { toSetoid := toSetoid✝¹, isOpen_setOf_rel := isOpen_setOf_rel✝¹ } { toSe …
                            -/
  | ⟨_, _⟩, ⟨_, _⟩, _ => by congr
                            /-
                              🎉 no goals
                            -/


/-- Construct a discrete quotient from a clopen set. -/
def ofIsClopen {A : Set X} (h : IsClopen A) : DiscreteQuotient X where
  toSetoid := ⟨fun x y => x ∈ A ↔ y ∈ A, fun _ => Iff.rfl, Iff.symm, Iff.trans⟩
                           /-
                             α : Type u_1
                             X : Type u_2
                             Y : Type u_3
                             Z : Type u_4
                             inst✝² : TopologicalSpace X
                             inst✝¹ : TopologicalSpace Y
                             inst✝ : TopologicalSpace Z
                             S : DiscreteQuotient X
                             A : Set X
                             h : IsClopen A
                             x : X
                             ⊢ IsOpen (setOf ({ r := fun x y => Iff (Membership.mem A x) (Membership.mem A  …
                           -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  isOpen_setOf_rel x := by by_cases hx : x ∈ A <;> simp [hx, h.1, h.2, ← compl_setOf]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem refl : ∀ x, S.toSetoid x x := S.refl'


theorem symm (x y : X) : S.toSetoid x y → S.toSetoid y x := S.symm'


theorem trans (x y z : X) : S.toSetoid x y → S.toSetoid y z → S.toSetoid x z := S.trans'


instance : CoeSort (DiscreteQuotient X) (Type _) :=
  ⟨fun S => Quotient S.toSetoid⟩


instance : TopologicalSpace S :=
  inferInstanceAs (TopologicalSpace (Quotient S.toSetoid))


/-- The projection from `X` to the given discrete quotient. -/
def proj : X → S := Quotient.mk''


theorem fiber_eq (x : X) : S.proj ⁻¹' {S.proj x} = setOf (S.toSetoid x) :=
  Set.ext fun _ => eq_comm.trans Quotient.eq''


theorem proj_surjective : Function.Surjective S.proj :=
  Quotient.mk''_surjective


theorem proj_isQuotientMap : IsQuotientMap S.proj :=
  isQuotientMap_quot_mk


@[deprecated (since := "2024-10-22")]
alias proj_quotientMap := proj_isQuotientMap


theorem proj_continuous : Continuous S.proj :=
  S.proj_isQuotientMap.continuous


instance : DiscreteTopology S :=
  singletons_open_iff_discrete.1 <| S.proj_surjective.forall.2 fun x => by
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      S : DiscreteQuotient X
      x : X
      ⊢ IsOpen (Singleton.singleton (S.proj x))
    -/
    rw [← S.proj_isQuotientMap.isOpen_preimage, fiber_eq]
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      S : DiscreteQuotient X
      x : X
      ⊢ IsOpen (setOf (S.toSetoid x))
    -/
    exact S.isOpen_setOf_rel _
    /-
      🎉 no goals
    -/


theorem proj_isLocallyConstant : IsLocallyConstant S.proj :=
  (IsLocallyConstant.iff_continuous S.proj).2 S.proj_continuous


theorem isClopen_preimage (A : Set S) : IsClopen (S.proj ⁻¹' A) :=
  (isClopen_discrete A).preimage S.proj_continuous


theorem isOpen_preimage (A : Set S) : IsOpen (S.proj ⁻¹' A) :=
  (S.isClopen_preimage A).2


theorem isClosed_preimage (A : Set S) : IsClosed (S.proj ⁻¹' A) :=
  (S.isClopen_preimage A).1


theorem isClopen_setOf_rel (x : X) : IsClopen (setOf (S.toSetoid x)) := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    S : DiscreteQuotient X
    x : X
    ⊢ IsClopen (setOf (S.toSetoid x))
  -/
  rw [← fiber_eq]
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    S : DiscreteQuotient X
    x : X
    ⊢ IsClopen (Set.preimage S.proj (Singleton.singleton (S.proj x)))
  -/
  apply isClopen_preimage
  /-
    🎉 no goals
  -/


instance : Min (DiscreteQuotient X) :=
  ⟨fun S₁ S₂ => ⟨S₁.1 ⊓ S₂.1, fun x => (S₁.2 x).inter (S₂.2 x)⟩⟩


instance : SemilatticeInf (DiscreteQuotient X) :=
  Injective.semilatticeInf toSetoid toSetoid_injective fun _ _ => rfl


instance : OrderTop (DiscreteQuotient X) where
  top := ⟨⊤, fun _ => isOpen_univ⟩
                 /-
                   α : Type u_1
                   X : Type u_2
                   Y : Type u_3
                   Z : Type u_4
                   inst✝² : TopologicalSpace X
                   inst✝¹ : TopologicalSpace Y
                   inst✝ : TopologicalSpace Z
                   S a : DiscreteQuotient X
                   ⊢ LE.le a Top.top
                 -/
  le_top a := by tauto
                 /-
                   🎉 no goals
                 -/


instance : Inhabited (DiscreteQuotient X) := ⟨⊤⟩


instance inhabitedQuotient [Inhabited X] : Inhabited S := ⟨S.proj default⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: add instances about `Nonempty (Quot _)`/`Nonempty (Quotient _)`

instance [Nonempty X] : Nonempty S := Nonempty.map S.proj ‹_›


/-- The quotient by `⊤ : DiscreteQuotient X` is a `Subsingleton`. -/
instance : Subsingleton (⊤ : DiscreteQuotient X) where
              /-
                α : Type u_1
                X : Type u_2
                Y : Type u_3
                Z : Type u_4
                inst✝² : TopologicalSpace X
                inst✝¹ : TopologicalSpace Y
                inst✝ : TopologicalSpace Z
                S : DiscreteQuotient X
                ⊢ ∀ (a b : Quotient Top.top.toSetoid), Eq a b
              -/
  allEq := by rintro ⟨_⟩ ⟨_⟩; exact Quotient.sound trivial
                              /-
                                🎉 no goals
                              -/


/-- Comap a discrete quotient along a continuous map. -/
def comap (S : DiscreteQuotient Y) : DiscreteQuotient X where
  toSetoid := Setoid.comap f S.1
  isOpen_setOf_rel _ := (S.2 _).preimage f.continuous


@[simp]
theorem comap_id : S.comap (ContinuousMap.id X) = S := rfl


@[simp]
theorem comap_comp (S : DiscreteQuotient Z) : S.comap (g.comp f) = (S.comap g).comap f :=
  rfl


@[mono]
                                                                                        /-
                                                                                          X : Type u_2
                                                                                          Y : Type u_3
                                                                                          inst✝¹ : TopologicalSpace X
                                                                                          inst✝ : TopologicalSpace Y
                                                                                          f : ContinuousMap X Y
                                                                                          A B : DiscreteQuotient Y
                                                                                          h : LE.le A B
                                                                                          ⊢ LE.le (DiscreteQuotient.comap f A) (DiscreteQuotient.comap f B)
                                                                                        -/
theorem comap_mono {A B : DiscreteQuotient Y} (h : A ≤ B) : A.comap f ≤ B.comap f := by tauto
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


/-- The map induced by a refinement of a discrete quotient. -/
def ofLE (h : A ≤ B) : A → B :=
  Quotient.map' id h


@[simp]
theorem ofLE_refl : ofLE (le_refl A) = id := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    A : DiscreteQuotient X
    ⊢ Eq (DiscreteQuotient.ofLE ⋯) id
  -/
  ext ⟨⟩
  /-
    case h.mk
    X : Type u_2
    inst✝ : TopologicalSpace X
    A : DiscreteQuotient X
    x✝ : Quotient A.toSetoid
    a✝ : X
    ⊢ Eq (DiscreteQuotient.ofLE ⋯ (Quot.mk (⇑A.toSetoid) a✝)) (id (Quot.mk (⇑A.toS …
  -/
  rfl
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 X : Type u_2
                                                                 inst✝ : TopologicalSpace X
                                                                 A : DiscreteQuotient X
                                                                 a : Quotient A.toSetoid
                                                                 ⊢ Eq (DiscreteQuotient.ofLE ⋯ a) a
                                                               -/
theorem ofLE_refl_apply (a : A) : ofLE (le_refl A) a = a := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem ofLE_ofLE (h₁ : A ≤ B) (h₂ : B ≤ C) (x : A) :
    ofLE h₂ (ofLE h₁ x) = ofLE (h₁.trans h₂) x := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    A B C : DiscreteQuotient X
    h₁ : LE.le A B
    h₂ : LE.le B C
    x : Quotient A.toSetoid
    ⊢ Eq (DiscreteQuotient.ofLE h₂ (DiscreteQuotient.ofLE h₁ x)) (DiscreteQuotient …
  -/
  rcases x with ⟨⟩
  /-
    case mk
    X : Type u_2
    inst✝ : TopologicalSpace X
    A B C : DiscreteQuotient X
    h₁ : LE.le A B
    h₂ : LE.le B C
    x : Quotient A.toSetoid
    a✝ : X
    ⊢ Eq (DiscreteQuotient.ofLE h₂ (DiscreteQuotient.ofLE h₁ (Quot.mk (⇑A.toSetoid …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofLE_comp_ofLE (h₁ : A ≤ B) (h₂ : B ≤ C) : ofLE h₂ ∘ ofLE h₁ = ofLE (le_trans h₁ h₂) :=
  funext <| ofLE_ofLE _ _


theorem ofLE_continuous (h : A ≤ B) : Continuous (ofLE h) :=
  continuous_of_discreteTopology


@[simp]
theorem ofLE_proj (h : A ≤ B) (x : X) : ofLE h (A.proj x) = B.proj x :=
  Quotient.sound' (B.refl _)


@[simp]
theorem ofLE_comp_proj (h : A ≤ B) : ofLE h ∘ A.proj = B.proj :=
  funext <| ofLE_proj _


/-- When `X` is a locally connected space, there is an `OrderBot` instance on
`DiscreteQuotient X`. The bottom element is given by `connectedComponentSetoid X`
-/
instance [LocallyConnectedSpace X] : OrderBot (DiscreteQuotient X) where
  bot :=
    { toSetoid := connectedComponentSetoid X
      isOpen_setOf_rel := fun x => by
        /-
          α : Type u_1
          X : Type u_2
          Y : Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : TopologicalSpace Z
          S : DiscreteQuotient X
          inst✝ : LocallyConnectedSpace X
          x : X
          ⊢ IsOpen (setOf ((connectedComponentSetoid X) x))
        -/
        convert isOpen_connectedComponent (x := x)
        /-
          case h.e'_3
          α : Type u_1
          X : Type u_2
          Y : Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : TopologicalSpace Z
          S : DiscreteQuotient X
          inst✝ : LocallyConnectedSpace X
          x : X
          ⊢ Eq (setOf ((connectedComponentSetoid X) x)) (connectedComponent x)
        -/
        ext y
        /-
          case h.e'_3.h
          α : Type u_1
          X : Type u_2
          Y : Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace X
          inst✝² : TopologicalSpace Y
          inst✝¹ : TopologicalSpace Z
          S : DiscreteQuotient X
          inst✝ : LocallyConnectedSpace X
          x y : X
          ⊢ Iff (Membership.mem (setOf ((connectedComponentSetoid X) x)) y) (Membership. …
        -/
        simpa only [connectedComponentSetoid, ← connectedComponent_eq_iff_mem] using eq_comm }
        /-
          🎉 no goals
        -/
  bot_le S := fun x y (h : connectedComponent x = connectedComponent y) =>
    (S.isClopen_setOf_rel x).connectedComponent_subset (S.refl _) <| h.symm ▸ mem_connectedComponent


@[simp]
theorem proj_bot_eq [LocallyConnectedSpace X] {x y : X} :
    proj ⊥ x = proj ⊥ y ↔ connectedComponent x = connectedComponent y :=
  Quotient.eq''


                                                                                        /-
                                                                                          X : Type u_2
                                                                                          inst✝¹ : TopologicalSpace X
                                                                                          inst✝ : DiscreteTopology X
                                                                                          x y : X
                                                                                          ⊢ Iff (Eq (Bot.bot.proj x) (Bot.bot.proj y)) (Eq x y)
                                                                                        -/
theorem proj_bot_inj [DiscreteTopology X] {x y : X} : proj ⊥ x = proj ⊥ y ↔ x = y := by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


theorem proj_bot_injective [DiscreteTopology X] : Injective (⊥ : DiscreteQuotient X).proj :=
  fun _ _ => proj_bot_inj.1


theorem proj_bot_bijective [DiscreteTopology X] : Bijective (⊥ : DiscreteQuotient X).proj :=
  ⟨proj_bot_injective, proj_surjective _⟩


/-- Given `f : C(X, Y)`, `DiscreteQuotient.LEComap f A B` is defined as
`A ≤ B.comap f`. Mathematically this means that `f` descends to a morphism `A → B`. -/
def LEComap : Prop :=
  A ≤ B.comap f


theorem leComap_id : LEComap (.id X) A A := le_rfl


@[simp]
theorem leComap_id_iff : LEComap (ContinuousMap.id X) A A' ↔ A ≤ A' :=
  Iff.rfl


                                                                                    /-
                                                                                      X : Type u_2
                                                                                      Y : Type u_3
                                                                                      Z : Type u_4
                                                                                      inst✝² : TopologicalSpace X
                                                                                      inst✝¹ : TopologicalSpace Y
                                                                                      inst✝ : TopologicalSpace Z
                                                                                      f : ContinuousMap X Y
                                                                                      A : DiscreteQuotient X
                                                                                      B : DiscreteQuotient Y
                                                                                      g : ContinuousMap Y Z
                                                                                      C : DiscreteQuotient Z
                                                                                      ⊢ DiscreteQuotient.LEComap g B C → DiscreteQuotient.LEComap f A B → DiscreteQu …
                                                                                    -/
theorem LEComap.comp : LEComap g B C → LEComap f A B → LEComap (g.comp f) A C := by tauto
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[mono]
theorem LEComap.mono (h : LEComap f A B) (hA : A' ≤ A) (hB : B ≤ B') : LEComap f A' B' :=
  hA.trans <| h.trans <| comap_mono _ hB


/-- Map a discrete quotient along a continuous map. -/
def map (f : C(X, Y)) (cond : LEComap f A B) : A → B := Quotient.map' f cond


theorem map_continuous (cond : LEComap f A B) : Continuous (map f cond) :=
  continuous_of_discreteTopology


@[simp]
theorem map_comp_proj (cond : LEComap f A B) : map f cond ∘ A.proj = B.proj ∘ f :=
  rfl


@[simp]
theorem map_proj (cond : LEComap f A B) (x : X) : map f cond (A.proj x) = B.proj (f x) :=
  rfl


@[simp]
                                                 /-
                                                   X : Type u_2
                                                   inst✝ : TopologicalSpace X
                                                   A : DiscreteQuotient X
                                                   ⊢ Eq (DiscreteQuotient.map (ContinuousMap.id X) ⋯) id
                                                 -/
theorem map_id : map _ (leComap_id A) = id := by ext ⟨⟩; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: figure out why `simpNF` says this is a bad `@[simp]` lemma
-- See https://github.com/leanprover-community/batteries/issues/365

theorem map_comp (h1 : LEComap g B C) (h2 : LEComap f A B) :
    map (g.comp f) (h1.comp h2) = map g h1 ∘ map f h2 := by
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    A : DiscreteQuotient X
    B : DiscreteQuotient Y
    g : ContinuousMap Y Z
    C : DiscreteQuotient Z
    h1 : DiscreteQuotient.LEComap g B C
    h2 : DiscreteQuotient.LEComap f A B
    ⊢ Eq (DiscreteQuotient.map (g.comp f) ⋯) (Function.comp (DiscreteQuotient.map  …
  -/
  ext ⟨⟩
  /-
    case h.mk
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : ContinuousMap X Y
    A : DiscreteQuotient X
    B : DiscreteQuotient Y
    g : ContinuousMap Y Z
    C : DiscreteQuotient Z
    h1 : DiscreteQuotient.LEComap g B C
    h2 : DiscreteQuotient.LEComap f A B
    x✝ : Quotient A.toSetoid
    a✝ : X
    ⊢ Eq (DiscreteQuotient.map (g.comp f) ⋯ (Quot.mk (⇑A.toSetoid) a✝)) (Function. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofLE_map (cond : LEComap f A B) (h : B ≤ B') (a : A) :
    ofLE h (map f cond a) = map f (cond.mono le_rfl h) a := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    A : DiscreteQuotient X
    B B' : DiscreteQuotient Y
    cond : DiscreteQuotient.LEComap f A B
    h : LE.le B B'
    a : Quotient A.toSetoid
    ⊢ Eq (DiscreteQuotient.ofLE h (DiscreteQuotient.map f cond a)) (DiscreteQuotie …
  -/
  rcases a with ⟨⟩
  /-
    case mk
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    A : DiscreteQuotient X
    B B' : DiscreteQuotient Y
    cond : DiscreteQuotient.LEComap f A B
    h : LE.le B B'
    a : Quotient A.toSetoid
    a✝ : X
    ⊢ Eq (DiscreteQuotient.ofLE h (DiscreteQuotient.map f cond (Quot.mk (⇑A.toSeto …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofLE_comp_map (cond : LEComap f A B) (h : B ≤ B') :
    ofLE h ∘ map f cond = map f (cond.mono le_rfl h) :=
  funext <| ofLE_map cond h


@[simp]
theorem map_ofLE (cond : LEComap f A B) (h : A' ≤ A) (c : A') :
    map f cond (ofLE h c) = map f (cond.mono h le_rfl) c := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    A A' : DiscreteQuotient X
    B : DiscreteQuotient Y
    cond : DiscreteQuotient.LEComap f A B
    h : LE.le A' A
    c : Quotient A'.toSetoid
    ⊢ Eq (DiscreteQuotient.map f cond (DiscreteQuotient.ofLE h c)) (DiscreteQuotie …
  -/
  rcases c with ⟨⟩
  /-
    case mk
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    A A' : DiscreteQuotient X
    B : DiscreteQuotient Y
    cond : DiscreteQuotient.LEComap f A B
    h : LE.le A' A
    c : Quotient A'.toSetoid
    a✝ : X
    ⊢ Eq (DiscreteQuotient.map f cond (DiscreteQuotient.ofLE h (Quot.mk (⇑A'.toSet …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_ofLE (cond : LEComap f A B) (h : A' ≤ A) :
    map f cond ∘ ofLE h = map f (cond.mono h le_rfl) :=
  funext <| map_ofLE cond h


theorem eq_of_forall_proj_eq [T2Space X] [CompactSpace X] [disc : TotallyDisconnectedSpace X]
    {x y : X} (h : ∀ Q : DiscreteQuotient X, Q.proj x = Q.proj y) : x = y := by
  rw [← mem_singleton_iff, ← connectedComponent_eq_singleton, connectedComponent_eq_iInter_isClopen,
    mem_iInter]
  /-
    X : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    disc : TotallyDisconnectedSpace X
    x y : X
    h : ∀ (Q : DiscreteQuotient X), Eq (Q.proj x) (Q.proj y)
    ⊢ ∀ (i : Subtype fun s => And (IsClopen s) (Membership.mem s y)), Membership.m …
  -/
  rintro ⟨U, hU1, hU2⟩
  /-
    case mk.intro
    X : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : CompactSpace X
    disc : TotallyDisconnectedSpace X
    x y : X
    h : ∀ (Q : DiscreteQuotient X), Eq (Q.proj x) (Q.proj y)
    U : Set X
    hU1 : IsClopen U
    hU2 : Membership.mem U y
    ⊢ Membership.mem (↑⟨U, ⋯⟩) x
  -/
  exact (Quotient.exact' (h (ofIsClopen hU1))).mpr hU2
  /-
    🎉 no goals
  -/


theorem fiber_subset_ofLE {A B : DiscreteQuotient X} (h : A ≤ B) (a : A) :
    A.proj ⁻¹' {a} ⊆ B.proj ⁻¹' {ofLE h a} := by
  /-
    X : Type u_2
    inst✝ : TopologicalSpace X
    A B : DiscreteQuotient X
    h : LE.le A B
    a : Quotient A.toSetoid
    ⊢ HasSubset.Subset (Set.preimage A.proj (Singleton.singleton a)) (Set.preimage …
  -/
  rcases A.proj_surjective a with ⟨a, rfl⟩
  /-
    case intro
    X : Type u_2
    inst✝ : TopologicalSpace X
    A B : DiscreteQuotient X
    h : LE.le A B
    a : X
    ⊢ HasSubset.Subset (Set.preimage A.proj (Singleton.singleton (A.proj a))) (Set …
  -/
  rw [fiber_eq, ofLE_proj, fiber_eq]
  /-
    case intro
    X : Type u_2
    inst✝ : TopologicalSpace X
    A B : DiscreteQuotient X
    h : LE.le A B
    a : X
    ⊢ HasSubset.Subset (setOf (A.toSetoid a)) (setOf (B.toSetoid a))
  -/
  exact fun _ h' => h h'
  /-
    🎉 no goals
  -/


theorem exists_of_compat [CompactSpace X] (Qs : (Q : DiscreteQuotient X) → Q)
    (compat : ∀ (A B : DiscreteQuotient X) (h : A ≤ B), ofLE h (Qs _) = Qs _) :
    ∃ x : X, ∀ Q : DiscreteQuotient X, Q.proj x = Qs _ := by
  have H₁ : ∀ Q₁ Q₂, Q₁ ≤ Q₂ → proj Q₁ ⁻¹' {Qs Q₁} ⊆ proj Q₂ ⁻¹' {Qs Q₂} := fun _ _ h => by
    rw [← compat _ _ h]
    exact fiber_subset_ofLE _ _
  obtain ⟨x, hx⟩ : Set.Nonempty (⋂ Q, proj Q ⁻¹' {Qs Q}) :=
    IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
      (fun Q : DiscreteQuotient X => Q.proj ⁻¹' {Qs _}) (directed_of_isDirected_ge H₁)
      (fun Q => (singleton_nonempty _).preimage Q.proj_surjective)
      (fun Q => (Q.isClosed_preimage {Qs _}).isCompact) fun Q => Q.isClosed_preimage _
  /-
    case intro
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    Qs : (Q : DiscreteQuotient X) → Quotient Q.toSetoid
    compat : ∀ (A B : DiscreteQuotient X) (h : LE.le A B), Eq (DiscreteQuotient.of …
    H₁ : ∀ (Q₁ Q₂ : DiscreteQuotient X), LE.le Q₁ Q₂ → HasSubset.Subset (Set.preim …
    x : X
    hx : Membership.mem (Set.iInter fun Q => Set.preimage Q.proj (Singleton.single …
    ⊢ Exists fun x => ∀ (Q : DiscreteQuotient X), Eq (Q.proj x) (Qs Q)
  -/
  exact ⟨x, mem_iInter.1 hx⟩
  /-
    🎉 no goals
  -/


/-- If `X` is a compact space, then any discrete quotient of `X` is finite. -/
instance [CompactSpace X] : Finite S := by
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    S : DiscreteQuotient X
    inst✝ : CompactSpace X
    ⊢ Finite (Quotient S.toSetoid)
  -/
  have : CompactSpace S := Quotient.compactSpace
  /-
    α : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    S : DiscreteQuotient X
    inst✝ : CompactSpace X
    this : CompactSpace (Quotient S.toSetoid)
    ⊢ Finite (Quotient S.toSetoid)
  -/
  rwa [← isCompact_univ_iff, isCompact_iff_finite, finite_univ_iff] at this
  /-
    🎉 no goals
  -/


open Classical in
/--
If `X` is a compact space, then we associate to any discrete quotient on `X` a finite set of
clopen subsets of `X`, given by the fibers of `proj`.

TODO: prove that these form a partition of `X`
-/
noncomputable def finsetClopens [CompactSpace X]
    (d : DiscreteQuotient X) : Finset (Clopens X) := have : Fintype d := Fintype.ofFinite _
  (Set.range (fun (x : d) ↦ ⟨_, d.isClopen_preimage {x}⟩) : Set (Clopens X)).toFinset


/-- A helper lemma to prove that `finsetClopens X` is injective, see `finsetClopens_inj`. -/
lemma comp_finsetClopens [CompactSpace X] :
    (Set.image (fun (t : Clopens X) ↦ t.carrier) ∘ Finset.toSet) ∘
      finsetClopens X = fun ⟨f, _⟩ ↦ f.classes := by
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    ⊢ Eq (Function.comp (Function.comp (Set.image fun t => t.carrier) Finset.toSet …
  -/
  ext d
  simp only [Setoid.classes, Set.mem_setOf_eq, Function.comp_apply,
    finsetClopens, Set.coe_toFinset, Set.mem_image, Set.mem_range,
    exists_exists_eq_and]
  /-
    case h.h
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    d : DiscreteQuotient X
    x✝ : Set X
    ⊢ Iff (Exists fun a => Eq (Set.preimage d.proj (Singleton.singleton a)) x✝) (E …
  -/
  constructor
    /-
      case h.h.mp
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      d : DiscreteQuotient X
      x✝ : Set X
      ⊢ (Exists fun a => Eq (Set.preimage d.proj (Singleton.singleton a)) x✝) → Exis …
    -/
  · refine fun ⟨y, h⟩ ↦ ⟨Quotient.out (s := d.toSetoid) y, ?_⟩
    /-
      case h.h.mp
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      d : DiscreteQuotient X
      x✝¹ : Set X
      x✝ : Exists fun a => Eq (Set.preimage d.proj (Singleton.singleton a)) x✝¹
      y : Quotient d.toSetoid
      h : Eq (Set.preimage d.proj (Singleton.singleton y)) x✝¹
      ⊢ Eq x✝¹ (setOf fun x => d.toSetoid x y.out)
    -/
    ext
    /-
      case h.h.mp.h
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      d : DiscreteQuotient X
      x✝² : Set X
      x✝¹ : Exists fun a => Eq (Set.preimage d.proj (Singleton.singleton a)) x✝²
      y : Quotient d.toSetoid
      h : Eq (Set.preimage d.proj (Singleton.singleton y)) x✝²
      x✝ : X
      ⊢ Iff (Membership.mem x✝² x✝) (Membership.mem (setOf fun x => d.toSetoid x y.o …
    -/
    simpa [← h] using Quotient.mk_eq_iff_out (s := d.toSetoid)
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : CompactSpace X
      d : DiscreteQuotient X
      x✝ : Set X
      ⊢ (Exists fun y => Eq x✝ (setOf fun x => d.toSetoid x y)) → Exists fun a => Eq …
    -/
  · exact fun ⟨y, h⟩ ↦ ⟨d.proj y, by ext; simp [h, proj]⟩
    /-
      🎉 no goals
    -/


/-- `finsetClopens X` is injective. -/
theorem finsetClopens_inj [CompactSpace X] :
    (finsetClopens X).Injective := by
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    ⊢ Function.Injective (DiscreteQuotient.finsetClopens X)
  -/
  apply Function.Injective.of_comp (f := Set.image (fun (t : Clopens X) ↦ t.carrier) ∘ Finset.toSet)
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    ⊢ Function.Injective (Function.comp (Function.comp (Set.image fun t => t.carri …
  -/
  rw [comp_finsetClopens]
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    ⊢ Function.Injective fun x => DiscreteQuotient.comp_finsetClopens.match_1 X (f …
  -/
  intro ⟨_, _⟩ ⟨_, _⟩ h
  /-
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    toSetoid✝¹ : Setoid X
    isOpen_setOf_rel✝¹ : ∀ (x : X), IsOpen (setOf (toSetoid✝¹ x))
    toSetoid✝ : Setoid X
    isOpen_setOf_rel✝ : ∀ (x : X), IsOpen (setOf (toSetoid✝ x))
    h : Eq ((fun x => DiscreteQuotient.comp_finsetClopens.match_1 X (fun x => Set  …
    ⊢ Eq { toSetoid := toSetoid✝¹, isOpen_setOf_rel := isOpen_setOf_rel✝¹ } { toSe …
  -/
  congr
  /-
    case e_toSetoid
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    toSetoid✝¹ : Setoid X
    isOpen_setOf_rel✝¹ : ∀ (x : X), IsOpen (setOf (toSetoid✝¹ x))
    toSetoid✝ : Setoid X
    isOpen_setOf_rel✝ : ∀ (x : X), IsOpen (setOf (toSetoid✝ x))
    h : Eq ((fun x => DiscreteQuotient.comp_finsetClopens.match_1 X (fun x => Set  …
    ⊢ Eq toSetoid✝¹ toSetoid✝
  -/
  rw [Setoid.classes_inj]
  /-
    case e_toSetoid
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    toSetoid✝¹ : Setoid X
    isOpen_setOf_rel✝¹ : ∀ (x : X), IsOpen (setOf (toSetoid✝¹ x))
    toSetoid✝ : Setoid X
    isOpen_setOf_rel✝ : ∀ (x : X), IsOpen (setOf (toSetoid✝ x))
    h : Eq ((fun x => DiscreteQuotient.comp_finsetClopens.match_1 X (fun x => Set  …
    ⊢ Eq toSetoid✝¹.classes toSetoid✝.classes
  -/
  exact h
  /-
    🎉 no goals
  -/


/--
The discrete quotients of a compact space are in bijection with a subtype of the type of
`Finset (Clopens X)`.

TODO: show that this is precisely those finsets of clopens which form a partition of `X`.
-/
noncomputable
def equivFinsetClopens [CompactSpace X] := Equiv.ofInjective _ (finsetClopens_inj X)


/-- Any locally constant function induces a discrete quotient. -/
def discreteQuotient : DiscreteQuotient X where
  toSetoid := .comap f ⊥
  isOpen_setOf_rel _ := f.isLocallyConstant _


/-- The (locally constant) function from the discrete quotient associated to a locally constant
function. -/
def lift : LocallyConstant f.discreteQuotient α :=
  ⟨fun a => Quotient.liftOn' a f fun _ _ => id, fun _ => isOpen_discrete _⟩


@[simp]
theorem lift_comp_proj : f.lift ∘ f.discreteQuotient.proj = f := rfl


