scoped[Topology] notation "I^" N => N → I


/-- The points in a cube with at least one projection equal to 0 or 1. -/
def boundary (N : Type*) : Set (I^N) :=
  {y | ∃ i, y i = 0 ∨ y i = 1}


/-- The forward direction of the homeomorphism
  between the cube $I^N$ and $I × I^{N\setminus\{j\}}$. -/
abbrev splitAt (i : N) : (I^N) ≃ₜ I × I^{ j // j ≠ i } :=
  funSplitAt I i


/-- The backward direction of the homeomorphism
  between the cube $I^N$ and $I × I^{N\setminus\{j\}}$. -/
abbrev insertAt (i : N) : (I × I^{ j // j ≠ i }) ≃ₜ I^N :=
  (funSplitAt I i).symm


theorem insertAt_boundary (i : N) {t₀ : I} {t}
    (H : (t₀ = 0 ∨ t₀ = 1) ∨ t ∈ boundary { j // j ≠ i }) : insertAt i ⟨t₀, t⟩ ∈ boundary N := by
  /-
    N : Type u_1
    inst✝ : DecidableEq N
    i : N
    t₀ : ↑unitInterval
    t : (Subtype fun j => Ne j i) → ↑unitInterval
    H : Or (Or (Eq t₀ 0) (Eq t₀ 1)) (Membership.mem (Cube.boundary (Subtype fun j  …
    ⊢ Membership.mem (Cube.boundary N) ((Cube.insertAt i) { fst := t₀, snd := t })
  -/
  obtain H | ⟨j, H⟩ := H
    /-
      case inl
      N : Type u_1
      inst✝ : DecidableEq N
      i : N
      t₀ : ↑unitInterval
      t : (Subtype fun j => Ne j i) → ↑unitInterval
      H : Or (Eq t₀ 0) (Eq t₀ 1)
      ⊢ Membership.mem (Cube.boundary N) ((Cube.insertAt i) { fst := t₀, snd := t })
    -/
  · use i; rwa [funSplitAt_symm_apply, dif_pos rfl]
           /-
             🎉 no goals
           -/
    /-
      case inr.intro
      N : Type u_1
      inst✝ : DecidableEq N
      i : N
      t₀ : ↑unitInterval
      t : (Subtype fun j => Ne j i) → ↑unitInterval
      j : Subtype fun j => Ne j i
      H : Or (Eq (t j) 0) (Eq (t j) 1)
      ⊢ Membership.mem (Cube.boundary N) ((Cube.insertAt i) { fst := t₀, snd := t })
    -/
  · use j; rwa [funSplitAt_symm_apply, dif_neg j.prop, Subtype.coe_eta]
           /-
             🎉 no goals
           -/


/-- The space of paths with both endpoints equal to a specified point `x : X`. -/
abbrev LoopSpace :=
  Path x x


scoped[Topology.Homotopy] notation "Ω" => LoopSpace


instance LoopSpace.inhabited : Inhabited (Path x x) :=
  ⟨Path.refl x⟩


/-- The `n`-dimensional generalized loops based at `x` in a space `X` are
  continuous functions `I^n → X` that sends the boundary to `x`.
  We allow an arbitrary indexing type `N` in place of `Fin n` here. -/
def GenLoop : Set C(I^N, X) :=
  {p | ∀ y ∈ Cube.boundary N, p y = x}


@[inherit_doc] scoped[Topology.Homotopy] notation "Ω^" => GenLoop


instance instFunLike : FunLike (Ω^ N X x) (I^N) X where
  coe f := f.1
                                                        /-
                                                          N : Type u_1
                                                          X : Type u_2
                                                          inst✝ : TopologicalSpace X
                                                          x : X
                                                          x✝² x✝¹ : ↑(GenLoop N X x)
                                                          f : (N → ↑unitInterval) → X
                                                          continuous_toFun✝¹ : Continuous f
                                                          property✝¹ : Membership.mem (GenLoop N X x) { toFun := f, continuous_toFun :=  …
                                                          g : (N → ↑unitInterval) → X
                                                          continuous_toFun✝ : Continuous g
                                                          property✝ : Membership.mem (GenLoop N X x) { toFun := g, continuous_toFun := c …
                                                          x✝ : Eq ((fun f => ⇑↑f) ⟨{ toFun := f, continuous_toFun := continuous_toFun✝¹  …
                                                          ⊢ Eq ⟨{ toFun := f, continuous_toFun := continuous_toFun✝¹ }, property✝¹⟩ ⟨{ t …
                                                        -/
  coe_injective' := fun ⟨⟨f, _⟩, _⟩ ⟨⟨g, _⟩, _⟩ _ => by congr
                                                        /-
                                                          🎉 no goals
                                                        -/


@[ext]
theorem ext (f g : Ω^ N X x) (H : ∀ y, f y = g y) : f = g :=
  DFunLike.coe_injective' (funext H)


@[simp]
theorem mk_apply (f : C(I^N, X)) (H y) : (⟨f, H⟩ : Ω^ N X x) y = f y :=
  rfl


instance instContinuousEval : ContinuousEval (Ω^ N X x) (I^N) X :=
  /-
    N : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    x : X
    ⊢ ∀ (g : ↑(GenLoop N X x)), Eq ⇑↑g ⇑g
  -/
  .of_continuous_forget continuous_subtype_val
  /-
    🎉 no goals
  -/


instance instContinuousEvalConst : ContinuousEvalConst (Ω^ N X x) (I^N) X := inferInstance


/-- Copy of a `GenLoop` with a new map from the unit cube equal to the old one.
  Useful to fix definitional equalities. -/
def copy (f : Ω^ N X x) (g : (I^N) → X) (h : g = f) : Ω^ N X x :=
                           /-
                             N : Type u_1
                             X : Type u_2
                             inst✝ : TopologicalSpace X
                             x : X
                             f : ↑(GenLoop N X x)
                             g : (N → ↑unitInterval) → X
                             h : Eq g ⇑f
                             ⊢ Membership.mem (GenLoop N X x) { toFun := g, continuous_toFun := ⋯ }
                           -/
  ⟨⟨g, h.symm ▸ f.1.2⟩, by convert f.2⟩
                           /-
                             🎉 no goals
                           -/

/- porting note: this now requires the `instFunLike` instance,
  so the instance is now put before `copy`. -/

theorem coe_copy (f : Ω^ N X x) {g : (I^N) → X} (h : g = f) : ⇑(copy f g h) = g :=
  rfl


theorem copy_eq (f : Ω^ N X x) {g : (I^N) → X} (h : g = f) : copy f g h = f := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    x : X
    f : ↑(GenLoop N X x)
    g : (N → ↑unitInterval) → X
    h : Eq g ⇑f
    ⊢ Eq (GenLoop.copy f g h) f
  -/
  ext x
  /-
    case H
    N : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    x✝ : X
    f : ↑(GenLoop N X x✝)
    g : (N → ↑unitInterval) → X
    h : Eq g ⇑f
    x : N → ↑unitInterval
    ⊢ Eq ((GenLoop.copy f g h) x) (f x)
  -/
  exact congr_fun h x
  /-
    🎉 no goals
  -/


theorem boundary (f : Ω^ N X x) : ∀ y ∈ Cube.boundary N, f y = x :=
  f.2


/-- The constant `GenLoop` at `x`. -/
def const : Ω^ N X x :=
  ⟨ContinuousMap.const _ x, fun _ _ => rfl⟩


@[simp]
theorem const_apply {t} : (@const N X _ x) t = x :=
  rfl


instance inhabited : Inhabited (Ω^ N X x) :=
  ⟨const⟩


/-- The "homotopic relative to boundary" relation between `GenLoop`s. -/
def Homotopic (f g : Ω^ N X x) : Prop :=
  f.1.HomotopicRel g.1 (Cube.boundary N)


@[refl]
theorem refl (f : Ω^ N X x) : Homotopic f f :=
  ContinuousMap.HomotopicRel.refl _


@[symm]
nonrec theorem symm (H : Homotopic f g) : Homotopic g f :=
  H.symm


@[trans]
nonrec theorem trans (H0 : Homotopic f g) (H1 : Homotopic g h) : Homotopic f h :=
  H0.trans H1


theorem equiv : Equivalence (@Homotopic N X _ x) :=
  ⟨Homotopic.refl, Homotopic.symm, Homotopic.trans⟩


instance setoid (N) (x : X) : Setoid (Ω^ N X x) :=
  ⟨Homotopic, equiv⟩


/-- Loop from a generalized loop by currying $I^N → X$ into $I → (I^{N\setminus\{j\}} → X)$. -/
@[simps]
def toLoop (i : N) (p : Ω^ N X x) : Ω (Ω^ { j // j ≠ i } X x) const where
  toFun t :=
    ⟨(p.val.comp (Cube.insertAt i)).curry t, fun y yH =>
      p.property (Cube.insertAt i (t, y)) (Cube.insertAt_boundary i <| Or.inr yH)⟩
                /-
                  N : Type u_1
                  X : Type u_2
                  inst✝¹ : TopologicalSpace X
                  x : X
                  inst✝ : DecidableEq N
                  i : N
                  p : ↑(GenLoop N X x)
                  ⊢ Eq ({ toFun := fun t => ⟨((↑p).comp ↑(Cube.insertAt i)).curry t, ⋯⟩, continu …
                -/
  source' := by ext t; refine p.property (Cube.insertAt i (0, t)) ⟨i, Or.inl ?_⟩; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                /-
                  N : Type u_1
                  X : Type u_2
                  inst✝¹ : TopologicalSpace X
                  x : X
                  inst✝ : DecidableEq N
                  i : N
                  p : ↑(GenLoop N X x)
                  ⊢ Eq ({ toFun := fun t => ⟨((↑p).comp ↑(Cube.insertAt i)).curry t, ⋯⟩, continu …
                -/
  target' := by ext t; refine p.property (Cube.insertAt i (1, t)) ⟨i, Or.inr ?_⟩; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/



theorem continuous_toLoop (i : N) : Continuous (@toLoop N X _ x _ i) :=
  Path.continuous_uncurry_iff.1 <|
    Continuous.subtype_mk
      (continuous_eval.comp <|
        Continuous.prodMap
          (ContinuousMap.continuous_curry.comp <|
            (ContinuousMap.continuous_precomp _).comp continuous_subtype_val)
          continuous_id)
      _


/-- Generalized loop from a loop by uncurrying $I → (I^{N\setminus\{j\}} → X)$ into $I^N → X$. -/
@[simps]
def fromLoop (i : N) (p : Ω (Ω^ { j // j ≠ i } X x) const) : Ω^ N X x :=
                                        /-
                                          N : Type u_1
                                          X : Type u_2
                                          inst✝¹ : TopologicalSpace X
                                          x : X
                                          inst✝ : DecidableEq N
                                          i : N
                                          p : LoopSpace (↑(GenLoop (Subtype fun j => Ne j i) X x)) GenLoop.const
                                          ⊢ Continuous Subtype.val
                                        -/
  ⟨(ContinuousMap.comp ⟨Subtype.val, by fun_prop⟩ p.toContinuousMap).uncurry.comp
                                        /-
                                          🎉 no goals
                                        -/
    (Cube.splitAt i),
    by
    /-
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p : LoopSpace (↑(GenLoop (Subtype fun j => Ne j i) X x)) GenLoop.const
      ⊢ Membership.mem (GenLoop N X x) (({ toFun := Subtype.val, continuous_toFun := …
    -/
    rintro y ⟨j, Hj⟩
    simp only [ContinuousMap.comp_apply, ContinuousMap.coe_coe,
      funSplitAt_apply, ContinuousMap.uncurry_apply, ContinuousMap.coe_mk,
      Function.uncurry_apply_pair]
    /-
      case intro
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p : LoopSpace (↑(GenLoop (Subtype fun j => Ne j i) X x)) GenLoop.const
      y : N → ↑unitInterval
      j : N
      Hj : Or (Eq (y j) 0) (Eq (y j) 1)
      ⊢ Eq (↑(p.toContinuousMap (y i)) fun j => y ↑j) x
    -/
    obtain rfl | Hne := eq_or_ne j i
      /-
        case intro.inl
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        y : N → ↑unitInterval
        j : N
        Hj : Or (Eq (y j) 0) (Eq (y j) 1)
        p : LoopSpace (↑(GenLoop (Subtype fun j_1 => Ne j_1 j) X x)) GenLoop.const
        ⊢ Eq (↑(p.toContinuousMap (y j)) fun j_1 => y ↑j_1) x
      -/
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
    · cases' Hj with Hj Hj <;> simp only [Hj, p.coe_toContinuousMap, p.source, p.target] <;> rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
      /-
        case intro.inr
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i : N
        p : LoopSpace (↑(GenLoop (Subtype fun j => Ne j i) X x)) GenLoop.const
        y : N → ↑unitInterval
        j : N
        Hj : Or (Eq (y j) 0) (Eq (y j) 1)
        Hne : Ne j i
        ⊢ Eq (↑(p.toContinuousMap (y i)) fun j => y ↑j) x
      -/
    · exact GenLoop.boundary _ _ ⟨⟨j, Hne⟩, Hj⟩⟩
      /-
        🎉 no goals
      -/


theorem continuous_fromLoop (i : N) : Continuous (@fromLoop N X _ x _ i) :=
  ((ContinuousMap.continuous_precomp _).comp <|
        ContinuousMap.continuous_uncurry.comp <|
          (ContinuousMap.continuous_postcomp _).comp continuous_induced_dom).subtype_mk
    _


theorem to_from (i : N) (p : Ω (Ω^ { j // j ≠ i } X x) const) : toLoop i (fromLoop i p) = p := by
  simp_rw [toLoop, fromLoop, ContinuousMap.comp_assoc,
    toContinuousMap_comp_symm, ContinuousMap.comp_id]
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p : LoopSpace (↑(GenLoop (Subtype fun j => Ne j i) X x)) GenLoop.const
    ⊢ Eq { toFun := fun t => ⟨({ toFun := Subtype.val, continuous_toFun := ⋯ }.com …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- The `n+1`-dimensional loops are in bijection with the loops in the space of
  `n`-dimensional loops with base point `const`.
  We allow an arbitrary indexing type `N` in place of `Fin n` here. -/
@[simps]
def loopHomeo (i : N) : Ω^ N X x ≃ₜ Ω (Ω^ { j // j ≠ i } X x) const where
  toFun := toLoop i
  invFun := fromLoop i
                   /-
                     N : Type u_1
                     X : Type u_2
                     inst✝¹ : TopologicalSpace X
                     x : X
                     inst✝ : DecidableEq N
                     i : N
                     p : ↑(GenLoop N X x)
                     ⊢ Eq (GenLoop.fromLoop i (GenLoop.toLoop i p)) p
                   -/
  left_inv p := by ext; exact congr_arg p (by dsimp; exact Equiv.apply_symm_apply _ _)
                        /-
                          🎉 no goals
                        -/
  right_inv := to_from i
  continuous_toFun := continuous_toLoop i
  continuous_invFun := continuous_fromLoop i


theorem toLoop_apply (i : N) {p : Ω^ N X x} {t} {tn} :
    toLoop i p t tn = p (Cube.insertAt i ⟨t, tn⟩) :=
  rfl


theorem fromLoop_apply (i : N) {p : Ω (Ω^ { j // j ≠ i } X x) const} {t : I^N} :
    fromLoop i p t = p (t i) (Cube.splitAt i t).snd :=
  rfl


/-- Composition with `Cube.insertAt` as a continuous map. -/
abbrev cCompInsert (i : N) : C(C(I^N, X), C(I × I^{ j // j ≠ i }, X)) :=
  ⟨fun f => f.comp (Cube.insertAt i),
    (toContinuousMap <| Cube.insertAt i).continuous_precomp⟩


/-- A homotopy between `n+1`-dimensional loops `p` and `q` constant on the boundary
  seen as a homotopy between two paths in the space of `n`-dimensional paths. -/
def homotopyTo (i : N) {p q : Ω^ N X x} (H : p.1.HomotopyRel q.1 (Cube.boundary N)) :
    C(I × I, C(I^{ j // j ≠ i }, X)) :=
  ((⟨_, ContinuousMap.continuous_curry⟩ : C(_, _)).comp <|
      (cCompInsert i).comp H.toContinuousMap.curry).uncurry

-- porting note: `@[simps]` generates this lemma but it's named `homotopyTo_apply_apply` instead

theorem homotopyTo_apply (i : N) {p q : Ω^ N X x} (H : p.1.HomotopyRel q.1 <| Cube.boundary N)
    (t : I × I) (tₙ : I^{ j // j ≠ i }) :
    homotopyTo i H t tₙ = H (t.fst, Cube.insertAt i (t.snd, tₙ)) :=
  rfl


theorem homotopicTo (i : N) {p q : Ω^ N X x} :
    Homotopic p q → (toLoop i p).Homotopic (toLoop i q) := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    ⊢ GenLoop.Homotopic p q → Path.Homotopic (GenLoop.toLoop i p) (GenLoop.toLoop  …
  -/
  refine Nonempty.map fun H => ⟨⟨⟨fun t => ⟨homotopyTo i H t, ?_⟩, ?_⟩, ?_, ?_⟩, ?_⟩
    /-
      case refine_1
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      t : Prod ↑unitInterval ↑unitInterval
      ⊢ Membership.mem (GenLoop (Subtype fun j => Ne j i) X x) ((GenLoop.homotopyTo  …
    -/
  · rintro y ⟨i, iH⟩
    /-
      case refine_1.intro
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i✝ : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      t : Prod ↑unitInterval ↑unitInterval
      y : (Subtype fun j => Ne j i✝) → ↑unitInterval
      i : Subtype fun j => Ne j i✝
      iH : Or (Eq (y i) 0) (Eq (y i) 1)
      ⊢ Eq (((GenLoop.homotopyTo i✝ H) t) y) x
    -/
    rw [homotopyTo_apply, H.eq_fst, p.2]
    /-
      case refine_1.intro.a
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i✝ : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      t : Prod ↑unitInterval ↑unitInterval
      y : (Subtype fun j => Ne j i✝) → ↑unitInterval
      i : Subtype fun j => Ne j i✝
      iH : Or (Eq (y i) 0) (Eq (y i) 1)
      ⊢ Membership.mem (Cube.boundary N) ((Cube.insertAt i✝) { fst := t.2, snd := y })
    -/
    all_goals apply Cube.insertAt_boundary; right; exact ⟨i, iH⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      ⊢ Continuous fun t => ⟨(GenLoop.homotopyTo i H) t, ⋯⟩
    -/
  · continuity
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
    ⊢ ∀ (x_1 : ↑unitInterval), Eq ({ toFun := fun t => ⟨(GenLoop.homotopyTo i H) t …
  -/
  iterate 2 intro; ext; erw [homotopyTo_apply, toLoop_apply]; swap
    /-
      case refine_3.H
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      x✝ : ↑unitInterval
      y✝ : (Subtype fun j => Ne j i) → ↑unitInterval
      ⊢ Eq (H { fst := { fst := 0, snd := x✝ }.1, snd := (Cube.insertAt i) { fst :=  …
    -/
  · apply H.apply_zero
    /-
      🎉 no goals
    -/
    /-
      case refine_4.H
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
      x✝ : ↑unitInterval
      y✝ : (Subtype fun j => Ne j i) → ↑unitInterval
      ⊢ Eq (H { fst := { fst := 1, snd := x✝ }.1, snd := (Cube.insertAt i) { fst :=  …
    -/
  · apply H.apply_one
    /-
      🎉 no goals
    -/
  /-
    case refine_5
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
    ⊢ ∀ (t x_1 : ↑unitInterval), Membership.mem (Insert.insert 0 (Singleton.single …
  -/
  intro t y yH
  /-
    case refine_5
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
    t y : ↑unitInterval
    yH : Membership.mem (Insert.insert 0 (Singleton.singleton 1)) y
    ⊢ Eq ({ toFun := fun x_1 => { toFun := fun t => ⟨(GenLoop.homotopyTo i H) t, ⋯ …
  -/
  ext; erw [homotopyTo_apply]
  /-
    case refine_5.H
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
    t y : ↑unitInterval
    yH : Membership.mem (Insert.insert 0 (Singleton.singleton 1)) y
    y✝ : (Subtype fun j => Ne j i) → ↑unitInterval
    ⊢ Eq (H { fst := { fst := t, snd := y }.1, snd := (Cube.insertAt i) { fst := { …
  -/
  apply H.eq_fst; use i
  /-
    case h
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : (↑p).HomotopyRel (↑q) (Cube.boundary N)
    t y : ↑unitInterval
    yH : Membership.mem (Insert.insert 0 (Singleton.singleton 1)) y
    y✝ : (Subtype fun j => Ne j i) → ↑unitInterval
    ⊢ Or (Eq ((Cube.insertAt i) { fst := { fst := t, snd := y }.2, snd := y✝ } i)  …
  -/
  rw [funSplitAt_symm_apply, dif_pos rfl]; exact yH
                                           /-
                                             🎉 no goals
                                           -/


/-- The converse to `GenLoop.homotopyTo`: a homotopy between two loops in the space of
  `n`-dimensional loops can be seen as a homotopy between two `n+1`-dimensional paths. -/
@[simps!] def homotopyFrom (i : N) {p q : Ω^ N X x} (H : (toLoop i p).Homotopy (toLoop i q)) :
    C(I × I^N, X) :=
  (ContinuousMap.comp ⟨_, ContinuousMap.continuous_uncurry⟩
                                               /-
                                                 N : Type u_1
                                                 X : Type u_2
                                                 inst✝¹ : TopologicalSpace X
                                                 x : X
                                                 inst✝ : DecidableEq N
                                                 i : N
                                                 p q : ↑(GenLoop N X x)
                                                 H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
                                                 ⊢ Continuous Subtype.val
                                               -/
          (ContinuousMap.comp ⟨Subtype.val, by continuity⟩ H.toContinuousMap).curry).uncurry.comp <|
                                               /-
                                                 🎉 no goals
                                               -/
    (ContinuousMap.id I).prodMap (Cube.splitAt i)


theorem homotopicFrom (i : N) {p q : Ω^ N X x} :
    (toLoop i p).Homotopic (toLoop i q) → Homotopic p q := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    ⊢ Path.Homotopic (GenLoop.toLoop i p) (GenLoop.toLoop i q) → GenLoop.Homotopic …
  -/
  refine Nonempty.map fun H => ⟨⟨homotopyFrom i H, ?_, ?_⟩, ?_⟩
  /-
    case refine_1
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    p q : ↑(GenLoop N X x)
    H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
    ⊢ ∀ (x_1 : N → ↑unitInterval), Eq ((GenLoop.homotopyFrom i H).toFun { fst := 0 …
  -/
  pick_goal 3
    /-
      case refine_3
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
      ⊢ ∀ (t : ↑unitInterval) (x_1 : N → ↑unitInterval), Membership.mem (Cube.bounda …
    -/
  · rintro t y ⟨j, jH⟩
    /-
      case refine_3.intro
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
      t : ↑unitInterval
      y : N → ↑unitInterval
      j : N
      jH : Or (Eq (y j) 0) (Eq (y j) 1)
      ⊢ Eq ({ toFun := fun x_1 => { toContinuousMap := GenLoop.homotopyFrom i H, map …
    -/
    erw [homotopyFrom_apply]
    /-
      case refine_3.intro
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      p q : ↑(GenLoop N X x)
      H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
      t : ↑unitInterval
      y : N → ↑unitInterval
      j : N
      jH : Or (Eq (y j) 0) (Eq (y j) 1)
      ⊢ Eq (Function.uncurry (fun x_1 y => Function.uncurry (fun x_2 y => ↑(H { fst  …
    -/
    obtain rfl | h := eq_or_ne j i
      /-
        case refine_3.intro.inl
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        p q : ↑(GenLoop N X x)
        t : ↑unitInterval
        y : N → ↑unitInterval
        j : N
        jH : Or (Eq (y j) 0) (Eq (y j) 1)
        H : Path.Homotopy (GenLoop.toLoop j p) (GenLoop.toLoop j q)
        ⊢ Eq (Function.uncurry (fun x_1 y => Function.uncurry (fun x_2 y => ↑(H { fst  …
      -/
    · simp only [Prod.map_apply, id_eq, funSplitAt_apply, Function.uncurry_apply_pair]
      /-
        case refine_3.intro.inl
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        p q : ↑(GenLoop N X x)
        t : ↑unitInterval
        y : N → ↑unitInterval
        j : N
        jH : Or (Eq (y j) 0) (Eq (y j) 1)
        H : Path.Homotopy (GenLoop.toLoop j p) (GenLoop.toLoop j q)
        ⊢ Eq (↑(H { fst := t, snd := y j }) fun j_1 => y ↑j_1) (↑p y)
      -/
      rw [H.eq_fst]
      /-
        case refine_3.intro.inl
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        p q : ↑(GenLoop N X x)
        t : ↑unitInterval
        y : N → ↑unitInterval
        j : N
        jH : Or (Eq (y j) 0) (Eq (y j) 1)
        H : Path.Homotopy (GenLoop.toLoop j p) (GenLoop.toLoop j q)
        ⊢ Eq (↑((GenLoop.toLoop j p).toContinuousMap (y j)) fun j_1 => y ↑j_1) (↑p y)
      -/
      exacts [congr_arg p ((Cube.splitAt j).left_inv _), jH]
      /-
        🎉 no goals
      -/
      /-
        case refine_3.intro.inr
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i : N
        p q : ↑(GenLoop N X x)
        H : Path.Homotopy (GenLoop.toLoop i p) (GenLoop.toLoop i q)
        t : ↑unitInterval
        y : N → ↑unitInterval
        j : N
        jH : Or (Eq (y j) 0) (Eq (y j) 1)
        h : Ne j i
        ⊢ Eq (Function.uncurry (fun x_1 y => Function.uncurry (fun x_2 y => ↑(H { fst  …
      -/
    · rw [p.2 _ ⟨j, jH⟩]; apply boundary; exact ⟨⟨j, h⟩, jH⟩
                                          /-
                                            🎉 no goals
                                          -/
  all_goals
    intro
    apply (homotopyFrom_apply _ _ _).trans
    simp only [Prod.map_apply, id_eq, funSplitAt_apply,
      Function.uncurry_apply_pair, ContinuousMap.HomotopyWith.apply_zero,
      ContinuousMap.HomotopyWith.apply_one, ne_eq, Path.coe_toContinuousMap, toLoop_apply_coe,
      ContinuousMap.curry_apply, ContinuousMap.comp_apply]
    first
    | apply congr_arg p
    | apply congr_arg q
    apply (Cube.splitAt i).left_inv


/-- Concatenation of two `GenLoop`s along the `i`th coordinate. -/
def transAt (i : N) (f g : Ω^ N X x) : Ω^ N X x :=
  copy (fromLoop i <| (toLoop i f).trans <| toLoop i g)
    (fun t => if (t i : ℝ) ≤ 1 / 2
      then f (Function.update t i <| Set.projIcc 0 1 zero_le_one (2 * t i))
      else g (Function.update t i <| Set.projIcc 0 1 zero_le_one (2 * t i - 1)))
    (by
      /-
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i : N
        f g : ↑(GenLoop N X x)
        ⊢ Eq (fun t => ite (LE.le (↑(t i)) (1 / 2)) (f (Function.update t i (Set.projI …
      -/
      ext1; symm
      dsimp only [Path.trans, fromLoop, Path.coe_mk_mk, Function.comp_apply, mk_apply,
        ContinuousMap.comp_apply, ContinuousMap.coe_coe, funSplitAt_apply,
        ContinuousMap.uncurry_apply, ContinuousMap.coe_mk, Function.uncurry_apply_pair]
      /-
        case h
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i : N
        f g : ↑(GenLoop N X x)
        x✝ : N → ↑unitInterval
        ⊢ Eq (↑(ite (LE.le (↑(x✝ i)) (1 / 2)) (Path.extend (GenLoop.toLoop i f) (HMul. …
      -/
      split_ifs
        /-
          case pos
          N : Type u_1
          X : Type u_2
          inst✝¹ : TopologicalSpace X
          x : X
          inst✝ : DecidableEq N
          i : N
          f g : ↑(GenLoop N X x)
          x✝ : N → ↑unitInterval
          h✝ : LE.le (↑(x✝ i)) (1 / 2)
          ⊢ Eq (↑(Path.extend (GenLoop.toLoop i f) (HMul.hMul 2 ↑(x✝ i))) fun j => x✝ ↑j …
        -/
      · show f _ = _; congr 1
                      /-
                        🎉 no goals
                      -/
        /-
          case neg
          N : Type u_1
          X : Type u_2
          inst✝¹ : TopologicalSpace X
          x : X
          inst✝ : DecidableEq N
          i : N
          f g : ↑(GenLoop N X x)
          x✝ : N → ↑unitInterval
          h✝ : Not (LE.le (↑(x✝ i)) (1 / 2))
          ⊢ Eq (↑(Path.extend (GenLoop.toLoop i g) (HSub.hSub (HMul.hMul 2 ↑(x✝ i)) 1))  …
        -/
      · show g _ = _; congr 1)
                      /-
                        🎉 no goals
                      -/


/-- Reversal of a `GenLoop` along the `i`th coordinate. -/
def symmAt (i : N) (f : Ω^ N X x) : Ω^ N X x :=
  (copy (fromLoop i (toLoop i f).symm) fun t => f fun j => if j = i then σ (t i) else t j) <| by
    /-
      N : Type u_1
      X : Type u_2
      inst✝¹ : TopologicalSpace X
      x : X
      inst✝ : DecidableEq N
      i : N
      f : ↑(GenLoop N X x)
      ⊢ Eq (fun t => f fun j => ite (Eq j i) (unitInterval.symm (t i)) (t j)) ⇑(GenL …
    -/
    ext1; change _ = f _; congr; ext1; simp
                                       /-
                                         🎉 no goals
                                       -/


theorem transAt_distrib {i j : N} (h : i ≠ j) (a b c d : Ω^ N X x) :
    transAt i (transAt j a b) (transAt j c d) = transAt j (transAt i a c) (transAt i b d) := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Ne i j
    a b c d : ↑(GenLoop N X x)
    ⊢ Eq (GenLoop.transAt i (GenLoop.transAt j a b) (GenLoop.transAt j c d)) (GenL …
  -/
  ext; simp_rw [transAt, coe_copy, Function.update_apply, if_neg h, if_neg h.symm]
  /-
    case H
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Ne i j
    a b c d : ↑(GenLoop N X x)
    y✝ : N → ↑unitInterval
    ⊢ Eq (ite (LE.le (↑(y✝ i)) (1 / 2)) (ite (LE.le (↑(y✝ j)) (1 / 2)) (a (Functio …
  -/
  split_ifs <;>
      /-
        case pos
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i j : N
        h : Ne i j
        a b c d : ↑(GenLoop N X x)
        y✝ : N → ↑unitInterval
        h✝¹ : LE.le (↑(y✝ i)) (1 / 2)
        h✝ : LE.le (↑(y✝ j)) (1 / 2)
        ⊢ Eq (a (Function.update (Function.update y✝ i (Set.projIcc 0 1 GenLoop.transA …
      -/
      /-
        case pos.h.e_6.h.h
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i j : N
        h : Ne i j
        a b c d : ↑(GenLoop N X x)
        y✝ : N → ↑unitInterval
        h✝¹ : LE.le (↑(y✝ i)) (1 / 2)
        h✝ : LE.le (↑(y✝ j)) (1 / 2)
        x✝ : N
        ⊢ Eq (ite (Eq x✝ j) (Set.projIcc 0 1 GenLoop.transAt.proof_2 (HMul.hMul 2 ↑(y✝ …
      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
      /-
        case neg.h.e_6.h.h
        N : Type u_1
        X : Type u_2
        inst✝¹ : TopologicalSpace X
        x : X
        inst✝ : DecidableEq N
        i j : N
        h : Ne i j
        a b c d : ↑(GenLoop N X x)
        y✝ : N → ↑unitInterval
        h✝¹ : Not (LE.le (↑(y✝ i)) (1 / 2))
        h✝ : Not (LE.le (↑(y✝ j)) (1 / 2))
        x✝ : N
        ⊢ Eq (ite (Eq x✝ j) (Set.projIcc 0 1 GenLoop.transAt.proof_2 (HSub.hSub (HMul. …
      -/
      apply ite_ite_comm; rintro rfl; exact h.symm
                                      /-
                                        🎉 no goals
                                      -/


theorem fromLoop_trans_toLoop {i : N} {p q : Ω^ N X x} :
    fromLoop i ((toLoop i p).trans <| toLoop i q) = transAt i p q :=
  (copy_eq _ _).symm


theorem fromLoop_symm_toLoop {i : N} {p : Ω^ N X x} : fromLoop i (toLoop i p).symm = symmAt i p :=
  (copy_eq _ _).symm


/-- The `n`th homotopy group at `x` defined as the quotient of `Ω^n x` by the
  `GenLoop.Homotopic` relation. -/
def HomotopyGroup (N X : Type*) [TopologicalSpace X] (x : X) : Type _ :=
  Quotient (GenLoop.Homotopic.setoid N x)

-- Porting note: in Lean 3 this instance was derived

instance : Inhabited (HomotopyGroup N X x) :=
  inferInstanceAs <| Inhabited <| Quotient (GenLoop.Homotopic.setoid N x)


/-- Equivalence between the homotopy group of X and the fundamental group of
  `Ω^{j // j ≠ i} x`. -/
def homotopyGroupEquivFundamentalGroup (i : N) :
    HomotopyGroup N X x ≃ FundamentalGroup (Ω^ { j // j ≠ i } X x) const := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    ⊢ Equiv (HomotopyGroup N X x) (FundamentalGroup (↑(GenLoop (Subtype fun j => N …
  -/
  refine Equiv.trans ?_ (CategoryTheory.Groupoid.isoEquivHom _ _).symm
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    ⊢ Equiv (HomotopyGroup N X x) (Quiver.Hom { as := GenLoop.const } { as := GenL …
  -/
  apply Quotient.congr (loopHomeo i).toEquiv
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i : N
    ⊢ ∀ (a₁ a₂ : ↑(GenLoop N X x)), Iff ((GenLoop.Homotopic.setoid N x) a₁ a₂) ((P …
  -/
  exact fun p q => ⟨homotopicTo i, homotopicFrom i⟩
  /-
    🎉 no goals
  -/


/-- Homotopy group of finite index. -/
abbrev HomotopyGroup.Pi (n) (X : Type*) [TopologicalSpace X] (x : X) :=
  HomotopyGroup (Fin n) _ x


scoped[Topology] notation "π_" => HomotopyGroup.Pi


/-- The 0-dimensional generalized loops based at `x` are in bijection with `X`. -/
def genLoopHomeoOfIsEmpty (N x) [IsEmpty N] : Ω^ N X x ≃ₜ X where
  toFun f := f 0
  invFun y := ⟨ContinuousMap.const _ y, fun _ ⟨i, _⟩ => isEmptyElim i⟩
                   /-
                     N✝ : Type u_1
                     X : Type u_2
                     inst✝² : TopologicalSpace X
                     x✝ : X
                     inst✝¹ : DecidableEq N✝
                     N : Type ?u.165949
                     x : X
                     inst✝ : IsEmpty N
                     f : ↑(GenLoop N X x)
                     ⊢ Eq ((fun y => ⟨ContinuousMap.const (N → ↑unitInterval) y, ⋯⟩) ((fun f => f 0 …
                   -/
  left_inv f := by ext; exact congr_arg f (Subsingleton.elim _ _)
                        /-
                          🎉 no goals
                        -/
  right_inv _ := rfl
  continuous_invFun := ContinuousMap.const'.2.subtype_mk _


/-- The homotopy "group" indexed by an empty type is in bijection with
  the path components of `X`, aka the `ZerothHomotopy`. -/
def homotopyGroupEquivZerothHomotopyOfIsEmpty (N x) [IsEmpty N] :
    HomotopyGroup N X x ≃ ZerothHomotopy X :=
  Quotient.congr (genLoopHomeoOfIsEmpty N x).toEquiv
    (by
      -- joined iff homotopic
      /-
        N✝ : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x✝ : X
        inst✝¹ : DecidableEq N✝
        N : Type ?u.209958
        x : X
        inst✝ : IsEmpty N
        ⊢ ∀ (a₁ a₂ : ↑(GenLoop N X x)), Iff ((GenLoop.Homotopic.setoid N x) a₁ a₂) ((p …
      -/
      intros a₁ a₂
      /-
        N✝ : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x✝ : X
        inst✝¹ : DecidableEq N✝
        N : Type ?u.209958
        x : X
        inst✝ : IsEmpty N
        a₁ a₂ : ↑(GenLoop N X x)
        ⊢ Iff ((GenLoop.Homotopic.setoid N x) a₁ a₂) ((pathSetoid X) ((genLoopHomeoOfI …
      -/
      constructor <;> rintro ⟨H⟩
      exacts
        [⟨{ toFun := fun t => H ⟨t, isEmptyElim⟩
            source' := (H.apply_zero _).trans (congr_arg a₁ <| Subsingleton.elim _ _)
            target' := (H.apply_one _).trans (congr_arg a₂ <| Subsingleton.elim _ _) }⟩,
        ⟨{  toFun := fun t0 => H t0.fst
            map_zero_left := fun _ => H.source.trans (congr_arg a₁ <| Subsingleton.elim _ _)
            map_one_left := fun _ => H.target.trans (congr_arg a₂ <| Subsingleton.elim _ _)
            prop' := fun _ _ ⟨i, _⟩ => isEmptyElim i }⟩])


/-- The 0th homotopy "group" is in bijection with `ZerothHomotopy`. -/
def HomotopyGroup.pi0EquivZerothHomotopy : π_ 0 X x ≃ ZerothHomotopy X :=
  homotopyGroupEquivZerothHomotopyOfIsEmpty (Fin 0) x


/-- The 1-dimensional generalized loops based at `x` are in bijection with loops at `x`. -/
def genLoopEquivOfUnique (N) [Unique N] : Ω^ N X x ≃ Ω X x where
  toFun p :=
                                       /-
                                         N✝ : Type u_1
                                         X : Type u_2
                                         inst✝² : TopologicalSpace X
                                         x : X
                                         inst✝¹ : DecidableEq N✝
                                         N : Type ?u.265304
                                         inst✝ : Unique N
                                         p : ↑(GenLoop N X x)
                                         ⊢ Continuous fun t => p fun x => t
                                       -/
    Path.mk ⟨fun t => p fun _ => t, by continuity⟩
                                       /-
                                         🎉 no goals
                                       -/
      (GenLoop.boundary _ (fun _ => 0) ⟨default, Or.inl rfl⟩)
      (GenLoop.boundary _ (fun _ => 1) ⟨default, Or.inr rfl⟩)
  invFun p :=
                                 /-
                                   N✝ : Type u_1
                                   X : Type u_2
                                   inst✝² : TopologicalSpace X
                                   x : X
                                   inst✝¹ : DecidableEq N✝
                                   N : Type ?u.265304
                                   inst✝ : Unique N
                                   p : LoopSpace X x
                                   ⊢ Continuous fun c => p (c Inhabited.default)
                                 -/
    ⟨⟨fun c => p (c default), by continuity⟩,
                                 /-
                                   🎉 no goals
                                 -/
      by
      /-
        N✝ : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x : X
        inst✝¹ : DecidableEq N✝
        N : Type ?u.265304
        inst✝ : Unique N
        p : LoopSpace X x
        ⊢ Membership.mem (GenLoop N X x) { toFun := fun c => p (c Inhabited.default),  …
      -/
      rintro y ⟨i, iH | iH⟩ <;> cases Unique.eq_default i <;> apply (congr_arg p iH).trans
      /-
        case intro.inl.refl
        N✝ : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x : X
        inst✝¹ : DecidableEq N✝
        N : Type ?u.265304
        inst✝ : Unique N
        p : LoopSpace X x
        y : N → ↑unitInterval
        iH : Eq (y Inhabited.default) 0
        ⊢ Eq (p 0) x
      -/
      exacts [p.source, p.target]⟩
      /-
        🎉 no goals
      -/
                   /-
                     N✝ : Type u_1
                     X : Type u_2
                     inst✝² : TopologicalSpace X
                     x : X
                     inst✝¹ : DecidableEq N✝
                     N : Type ?u.265304
                     inst✝ : Unique N
                     p : ↑(GenLoop N X x)
                     ⊢ Eq ((fun p => ⟨{ toFun := fun c => p (c Inhabited.default), continuous_toFun …
                   -/
  left_inv p := by ext y; exact congr_arg p (eq_const_of_unique y).symm
                          /-
                            🎉 no goals
                          -/
                    /-
                      N✝ : Type u_1
                      X : Type u_2
                      inst✝² : TopologicalSpace X
                      x : X
                      inst✝¹ : DecidableEq N✝
                      N : Type ?u.265304
                      inst✝ : Unique N
                      p : LoopSpace X x
                      ⊢ Eq ((fun p => { toFun := fun t => p fun x => t, continuous_toFun := ⋯, sourc …
                    -/
  right_inv p := by ext; rfl
                         /-
                           🎉 no goals
                         -/

/- TODO (?): deducing this from `homotopyGroupEquivFundamentalGroup` would require
  combination of `CategoryTheory.Functor.mapAut` and
  `FundamentalGroupoid.fundamentalGroupoidFunctor` applied to `genLoopHomeoOfIsEmpty`,
  with possibly worse defeq. -/

/-- The homotopy group at `x` indexed by a singleton is in bijection with the fundamental group,
  i.e. the loops based at `x` up to homotopy. -/
def homotopyGroupEquivFundamentalGroupOfUnique (N) [Unique N] :
    HomotopyGroup N X x ≃ FundamentalGroup X x := by
  /-
    N✝ : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N✝
    N : Type ?u.316915
    inst✝ : Unique N
    ⊢ Equiv (HomotopyGroup N X x) (FundamentalGroup X x)
  -/
  refine Equiv.trans ?_ (CategoryTheory.Groupoid.isoEquivHom _ _).symm
  /-
    N✝ : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N✝
    N : Type ?u.316915
    inst✝ : Unique N
    ⊢ Equiv (HomotopyGroup N X x) (Quiver.Hom { as := x } { as := x })
  -/
  refine Quotient.congr (genLoopEquivOfUnique N) ?_
  /-
    N✝ : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N✝
    N : Type ?u.316915
    inst✝ : Unique N
    ⊢ ∀ (a₁ a₂ : ↑(GenLoop N X x)), Iff ((GenLoop.Homotopic.setoid N x) a₁ a₂) ((P …
  -/
  intros a₁ a₂; constructor <;> rintro ⟨H⟩
  · exact
      ⟨{  toFun := fun tx => H (tx.fst, fun _ => tx.snd)
          map_zero_left := fun _ => H.apply_zero _
          map_one_left := fun _ => H.apply_one _
          prop' := fun t y iH => H.prop' _ _ ⟨default, iH⟩ }⟩
  refine
    ⟨⟨⟨⟨fun tx => H (tx.fst, tx.snd default), H.continuous.comp ?_⟩, fun y => ?_, fun y => ?_⟩, ?_⟩⟩
    /-
      case mpr.intro.refine_1
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      ⊢ Continuous fun tx => { fst := tx.1, snd := tx.2 Inhabited.default }
    -/
  · exact continuous_fst.prod_mk ((continuous_apply _).comp continuous_snd)
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.refine_2
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      y : N → ↑unitInterval
      ⊢ Eq ({ toFun := fun tx => H { fst := tx.1, snd := tx.2 Inhabited.default }, c …
    -/
  · exact (H.apply_zero _).trans (congr_arg a₁ (eq_const_of_unique y).symm)
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.refine_3
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      y : N → ↑unitInterval
      ⊢ Eq ({ toFun := fun tx => H { fst := tx.1, snd := tx.2 Inhabited.default }, c …
    -/
  · exact (H.apply_one _).trans (congr_arg a₂ (eq_const_of_unique y).symm)
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.refine_4
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      ⊢ ∀ (t : ↑unitInterval) (x_1 : N → ↑unitInterval), Membership.mem (Cube.bounda …
    -/
  · rintro t y ⟨i, iH⟩
    /-
      case mpr.intro.refine_4.intro
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      t : ↑unitInterval
      y : N → ↑unitInterval
      i : N
      iH : Or (Eq (y i) 0) (Eq (y i) 1)
      ⊢ Eq ({ toFun := fun x_1 => { toFun := fun tx => H { fst := tx.1, snd := tx.2  …
    -/
    cases Unique.eq_default i
    /-
      case mpr.intro.refine_4.intro.refl
      N✝ : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      x : X
      inst✝¹ : DecidableEq N✝
      N : Type ?u.316915
      inst✝ : Unique N
      a₁ a₂ : ↑(GenLoop N X x)
      H : ((genLoopEquivOfUnique N) a₁).Homotopy ((genLoopEquivOfUnique N) a₂)
      t : ↑unitInterval
      y : N → ↑unitInterval
      iH : Or (Eq (y Inhabited.default) 0) (Eq (y Inhabited.default) 1)
      ⊢ Eq ({ toFun := fun x_1 => { toFun := fun tx => H { fst := tx.1, snd := tx.2  …
    -/
    exact (H.eq_fst _ iH).trans (congr_arg a₁ (eq_const_of_unique y).symm)
    /-
      🎉 no goals
    -/


/-- The first homotopy group at `x` is in bijection with the fundamental group. -/
def HomotopyGroup.pi1EquivFundamentalGroup : π_ 1 X x ≃ FundamentalGroup X x :=
  homotopyGroupEquivFundamentalGroupOfUnique (Fin 1)


/-- Group structure on `HomotopyGroup N X x` for nonempty `N` (in particular `π_(n+1) X x`). -/
instance group (N) [DecidableEq N] [Nonempty N] : Group (HomotopyGroup N X x) :=
  (homotopyGroupEquivFundamentalGroup <| Classical.arbitrary N).group


/-- Group structure on `HomotopyGroup` obtained by pulling back path composition along the
  `i`th direction. The group structures for two different `i j : N` distribute over each
  other, and therefore are equal by the Eckmann-Hilton argument. -/
abbrev auxGroup (i : N) : Group (HomotopyGroup N X x) :=
  (homotopyGroupEquivFundamentalGroup i).group


theorem isUnital_auxGroup (i : N) :
    EckmannHilton.IsUnital (auxGroup i).mul (⟦const⟧ : HomotopyGroup N X x) where
  left_id := (auxGroup i).one_mul
  right_id := (auxGroup i).mul_one


theorem auxGroup_indep (i j : N) : (auxGroup i : Group (HomotopyGroup N X x)) = auxGroup j := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    ⊢ Eq (HomotopyGroup.auxGroup i) (HomotopyGroup.auxGroup j)
  -/
  by_cases h : i = j; · rw [h]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Not (Eq i j)
    ⊢ Eq (HomotopyGroup.auxGroup i) (HomotopyGroup.auxGroup j)
  -/
  refine Group.ext (EckmannHilton.mul (isUnital_auxGroup i) (isUnital_auxGroup j) ?_)
  /-
    case neg
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Not (Eq i j)
    ⊢ ∀ (a b c d : HomotopyGroup N X x), Eq (Mul.mul (Mul.mul a b) (Mul.mul c d))  …
  -/
  rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ ⟨d⟩
  /-
    case neg.mk.mk.mk.mk
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Not (Eq i j)
    a✝ : HomotopyGroup N X x
    a : ↑(GenLoop N X x)
    b✝ : HomotopyGroup N X x
    b : ↑(GenLoop N X x)
    c✝ : HomotopyGroup N X x
    c : ↑(GenLoop N X x)
    d✝ : HomotopyGroup N X x
    d : ↑(GenLoop N X x)
    ⊢ Eq (Mul.mul (Mul.mul (Quot.mk (⇑(GenLoop.Homotopic.setoid N x)) a) (Quot.mk  …
  -/
  change Quotient.mk' _ = _
  /-
    case neg.mk.mk.mk.mk
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    h : Not (Eq i j)
    a✝ : HomotopyGroup N X x
    a : ↑(GenLoop N X x)
    b✝ : HomotopyGroup N X x
    b : ↑(GenLoop N X x)
    c✝ : HomotopyGroup N X x
    c : ↑(GenLoop N X x)
    d✝ : HomotopyGroup N X x
    d : ↑(GenLoop N X x)
    ⊢ Eq (Quotient.mk' ((GenLoop.loopHomeo i).symm (Path.trans ((GenLoop.loopHomeo …
  -/
  apply congr_arg Quotient.mk'
  simp only [fromLoop_trans_toLoop, transAt_distrib h, coe_toEquiv, loopHomeo_apply,
    coe_symm_toEquiv, loopHomeo_symm_apply]


theorem transAt_indep {i} (j) (f g : Ω^ N X x) :
    (⟦transAt i f g⟧ : HomotopyGroup N X x) = ⟦transAt j f g⟧ := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f g : ↑(GenLoop N X x)
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.transAt i f g)) (Quo …
  -/
  simp_rw [← fromLoop_trans_toLoop]
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f g : ↑(GenLoop N X x)
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.fromLoop i (Path.tra …
  -/
  let m := fun (G) (_ : Group G) => ((· * ·) : G → G → G)
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f g : ↑(GenLoop N X x)
    m : (G : Type ?u.371251) → Group G → G → G → G := fun G x x1 x2 => HMul.hMul x …
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.fromLoop i (Path.tra …
  -/
  exact congr_fun₂ (congr_arg (m <| HomotopyGroup N X x) <| auxGroup_indep i j) ⟦g⟧ ⟦f⟧
  /-
    🎉 no goals
  -/


theorem symmAt_indep {i} (j) (f : Ω^ N X x) :
    (⟦symmAt i f⟧ : HomotopyGroup N X x) = ⟦symmAt j f⟧ := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f : ↑(GenLoop N X x)
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.symmAt i f)) (Quotie …
  -/
  simp_rw [← fromLoop_symm_toLoop]
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f : ↑(GenLoop N X x)
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.fromLoop i (Path.sym …
  -/
  let inv := fun (G) (_ : Group G) => ((·⁻¹) : G → G)
  /-
    N : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    x : X
    inst✝ : DecidableEq N
    i j : N
    f : ↑(GenLoop N X x)
    inv : (G : Type ?u.373110) → Group G → G → G := fun G x x_1 => Inv.inv x_1
    ⊢ Eq (Quotient.mk (GenLoop.Homotopic.setoid N x) (GenLoop.fromLoop i (Path.sym …
  -/
  exact congr_fun (congr_arg (inv <| HomotopyGroup N X x) <| auxGroup_indep i j) ⟦f⟧
  /-
    🎉 no goals
  -/


/-- Characterization of multiplicative identity -/
theorem one_def [Nonempty N] : (1 : HomotopyGroup N X x) = ⟦const⟧ :=
  rfl


/-- Characterization of multiplication -/
theorem mul_spec [Nonempty N] {i} {p q : Ω^ N X x} :
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: introduce `HomotopyGroup.mk` and remove defeq abuse.
    ((· * ·) : _ → _ → HomotopyGroup N X x) ⟦p⟧ ⟦q⟧ = ⟦transAt i q p⟧ := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p q : ↑(GenLoop N X x)
    ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (Quotient.mk (GenLoop.Homotopic.setoid N  …
  -/
  rw [transAt_indep (Classical.arbitrary N) q, ← fromLoop_trans_toLoop]
  /-
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p q : ↑(GenLoop N X x)
    ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (Quotient.mk (GenLoop.Homotopic.setoid N  …
  -/
  apply Quotient.sound
  /-
    case a
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p q : ↑(GenLoop N X x)
    ⊢ HasEquiv.Equiv ((GenLoop.loopHomeo (Classical.arbitrary N)).symm (Path.trans …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Characterization of multiplicative inverse -/
theorem inv_spec [Nonempty N] {i} {p : Ω^ N X x} :
    ((⟦p⟧)⁻¹ : HomotopyGroup N X x) = ⟦symmAt i p⟧ := by
  /-
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p : ↑(GenLoop N X x)
    ⊢ Eq (Inv.inv (Quotient.mk (GenLoop.Homotopic.setoid N x) p)) (Quotient.mk (Ge …
  -/
  rw [symmAt_indep (Classical.arbitrary N) p, ← fromLoop_symm_toLoop]
  /-
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p : ↑(GenLoop N X x)
    ⊢ Eq (Inv.inv (Quotient.mk (GenLoop.Homotopic.setoid N x) p)) (Quotient.mk (Ge …
  -/
  apply Quotient.sound
  /-
    case a
    N : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    x : X
    inst✝¹ : DecidableEq N
    inst✝ : Nonempty N
    i : N
    p : ↑(GenLoop N X x)
    ⊢ HasEquiv.Equiv ((GenLoop.loopHomeo (Classical.arbitrary N)).symm (Path.symm  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Multiplication on `HomotopyGroup N X x` is commutative for nontrivial `N`.
  In particular, multiplication on `π_(n+2)` is commutative. -/
instance commGroup [Nontrivial N] : CommGroup (HomotopyGroup N X x) :=
  let h := exists_ne (Classical.arbitrary N)
  @EckmannHilton.commGroup (HomotopyGroup N X x) _ 1 (isUnital_auxGroup <| Classical.choose h) _
    (by
      /-
        N : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x : X
        inst✝¹ : DecidableEq N
        inst✝ : Nontrivial N
        h : Exists fun y => Ne y (Classical.arbitrary N) := exists_ne (Classical.arbit …
        ⊢ ∀ (a b c d : HomotopyGroup N X x), Eq (Mul.mul (HMul.hMul a b) (HMul.hMul c  …
      -/
      rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ ⟨d⟩
      /-
        case mk.mk.mk.mk
        N : Type u_1
        X : Type u_2
        inst✝² : TopologicalSpace X
        x : X
        inst✝¹ : DecidableEq N
        inst✝ : Nontrivial N
        h : Exists fun y => Ne y (Classical.arbitrary N) := exists_ne (Classical.arbit …
        a✝ : HomotopyGroup N X x
        a : ↑(GenLoop N X x)
        b✝ : HomotopyGroup N X x
        b : ↑(GenLoop N X x)
        c✝ : HomotopyGroup N X x
        c : ↑(GenLoop N X x)
        d✝ : HomotopyGroup N X x
        d : ↑(GenLoop N X x)
        ⊢ Eq (Mul.mul (HMul.hMul (Quot.mk (⇑(GenLoop.Homotopic.setoid N x)) a) (Quot.m …
      -/
      apply congr_arg Quotient.mk'
      simp only [fromLoop_trans_toLoop, transAt_distrib <| Classical.choose_spec h, coe_toEquiv,
        loopHomeo_apply, coe_symm_toEquiv, loopHomeo_symm_apply])


