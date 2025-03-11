/-- A class encoding the concept that a space satisfies the Tietze extension property. -/
class TietzeExtension (Y : Type v) [TopologicalSpace Y] : Prop where
  exists_restrict_eq' {X : Type u} [TopologicalSpace X] [NormalSpace X] (s : Set X)
    (hs : IsClosed s) (f : C(s, Y)) : ∃ (g : C(X, Y)), g.restrict s = f


/-- **Tietze extension theorem** for `TietzeExtension` spaces, a version for a closed set. Let
`s` be a closed set in a normal topological space `X`. Let `f` be a continuous function
on `s` with values in a `TietzeExtension` space `Y`. Then there exists a continuous function
`g : C(X, Y)` such that `g.restrict s = f`. -/
theorem ContinuousMap.exists_restrict_eq (hs : IsClosed s) (f : C(s, Y)) :
    ∃ (g : C(X, Y)), g.restrict s = f :=
  TietzeExtension.exists_restrict_eq' s hs f


/-- **Tietze extension theorem** for `TietzeExtension` spaces. Let `e` be a closed embedding of a
nonempty topological space `X₁` into a normal topological space `X`. Let `f` be a continuous
function on `X₁` with values in a `TietzeExtension` space `Y`. Then there exists a
continuous function `g : C(X, Y)` such that `g ∘ e = f`. -/
theorem ContinuousMap.exists_extension (he : IsClosedEmbedding e) (f : C(X₁, Y)) :
    ∃ (g : C(X, Y)), g.comp ⟨e, he.continuous⟩ = f := by
  /-
    X₁ : Type u₁
    inst✝⁴ : TopologicalSpace X₁
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : NormalSpace X
    e : X₁ → X
    Y : Type v
    inst✝¹ : TopologicalSpace Y
    inst✝ : TietzeExtension Y
    he : Topology.IsClosedEmbedding e
    f : ContinuousMap X₁ Y
    ⊢ Exists fun g => Eq (g.comp { toFun := e, continuous_toFun := ⋯ }) f
  -/
  let e' : X₁ ≃ₜ Set.range e := Homeomorph.ofIsEmbedding _ he.isEmbedding
  /-
    X₁ : Type u₁
    inst✝⁴ : TopologicalSpace X₁
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : NormalSpace X
    e : X₁ → X
    Y : Type v
    inst✝¹ : TopologicalSpace Y
    inst✝ : TietzeExtension Y
    he : Topology.IsClosedEmbedding e
    f : ContinuousMap X₁ Y
    e' : Homeomorph X₁ ↑(Set.range e) := Homeomorph.ofIsEmbedding e ⋯
    ⊢ Exists fun g => Eq (g.comp { toFun := e, continuous_toFun := ⋯ }) f
  -/
  obtain ⟨g, hg⟩ := (f.comp e'.symm).exists_restrict_eq he.isClosed_range
  /-
    case intro
    X₁ : Type u₁
    inst✝⁴ : TopologicalSpace X₁
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : NormalSpace X
    e : X₁ → X
    Y : Type v
    inst✝¹ : TopologicalSpace Y
    inst✝ : TietzeExtension Y
    he : Topology.IsClosedEmbedding e
    f : ContinuousMap X₁ Y
    e' : Homeomorph X₁ ↑(Set.range e) := Homeomorph.ofIsEmbedding e ⋯
    g : ContinuousMap X Y
    hg : Eq (ContinuousMap.restrict (Set.range e) g) (f.comp ↑e'.symm)
    ⊢ Exists fun g => Eq (g.comp { toFun := e, continuous_toFun := ⋯ }) f
  -/
  exact ⟨g, by ext x; simpa using congr($(hg) ⟨e' x, x, rfl⟩)⟩
  /-
    🎉 no goals
  -/


/-- **Tietze extension theorem** for `TietzeExtension` spaces. Let `e` be a closed embedding of a
nonempty topological space `X₁` into a normal topological space `X`. Let `f` be a continuous
function on `X₁` with values in a `TietzeExtension` space `Y`. Then there exists a
continuous function `g : C(X, Y)` such that `g ∘ e = f`.

This version is provided for convenience and backwards compatibility. Here the composition is
phrased in terms of bare functions. -/
theorem ContinuousMap.exists_extension' (he : IsClosedEmbedding e) (f : C(X₁, Y)) :
    ∃ (g : C(X, Y)), g ∘ e = f :=
                                             /-
                                               X₁ : Type u₁
                                               inst✝⁴ : TopologicalSpace X₁
                                               X : Type u
                                               inst✝³ : TopologicalSpace X
                                               inst✝² : NormalSpace X
                                               e : X₁ → X
                                               Y : Type v
                                               inst✝¹ : TopologicalSpace Y
                                               inst✝ : TietzeExtension Y
                                               he : Topology.IsClosedEmbedding e
                                               f : ContinuousMap X₁ Y
                                               g : ContinuousMap X Y
                                               hg : Eq (g.comp { toFun := e, continuous_toFun := ⋯ }) f
                                               ⊢ Eq (Function.comp (⇑g) e) ⇑f
                                             -/
  f.exists_extension he |>.imp fun g hg ↦ by ext x; congrm($(hg) x)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- This theorem is not intended to be used directly because it is rare for a set alone to
satisfy `[TietzeExtension t]`. For example, `Metric.ball` in `ℝ` only satisfies it when
the radius is strictly positive, so finding this as an instance will fail.

Instead, it is intended to be used as a constructor for theorems about sets which *do* satisfy
`[TietzeExtension t]` under some hypotheses. -/
theorem ContinuousMap.exists_forall_mem_restrict_eq (hs : IsClosed s)
    {Y : Type v} [TopologicalSpace Y] (f : C(s, Y))
    {t : Set Y} (hf : ∀ x, f x ∈ t) [ht : TietzeExtension.{u, v} t] :
    ∃ (g : C(X, Y)), (∀ x, g x ∈ t) ∧ g.restrict s = f := by
  /-
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    Y : Type v
    inst✝ : TopologicalSpace Y
    f : ContinuousMap (↑s) Y
    t : Set Y
    hf : ∀ (x : ↑s), Membership.mem t (f x)
    ht : TietzeExtension ↑t
    ⊢ Exists fun g => And (∀ (x : X), Membership.mem t (g x)) (Eq (ContinuousMap.r …
  -/
  obtain ⟨g, hg⟩ := mk _ (map_continuous f |>.codRestrict hf) |>.exists_restrict_eq hs
  /-
    case intro
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    Y : Type v
    inst✝ : TopologicalSpace Y
    f : ContinuousMap (↑s) Y
    t : Set Y
    hf : ∀ (x : ↑s), Membership.mem t (f x)
    ht : TietzeExtension ↑t
    g : ContinuousMap X ↑t
    hg : Eq (ContinuousMap.restrict s g) { toFun := Set.codRestrict (⇑f) t hf, con …
    ⊢ Exists fun g => And (∀ (x : X), Membership.mem t (g x)) (Eq (ContinuousMap.r …
  -/
  exact ⟨comp ⟨Subtype.val, by continuity⟩ g, by simp, by ext x; congrm(($(hg) x : Y))⟩
  /-
    🎉 no goals
  -/


/-- This theorem is not intended to be used directly because it is rare for a set alone to
satisfy `[TietzeExtension t]`. For example, `Metric.ball` in `ℝ` only satisfies it when
the radius is strictly positive, so finding this as an instance will fail.

Instead, it is intended to be used as a constructor for theorems about sets which *do* satisfy
`[TietzeExtension t]` under some hypotheses. -/
theorem ContinuousMap.exists_extension_forall_mem (he : IsClosedEmbedding e)
    {Y : Type v} [TopologicalSpace Y] (f : C(X₁, Y))
    {t : Set Y} (hf : ∀ x, f x ∈ t) [ht : TietzeExtension.{u, v} t] :
    ∃ (g : C(X, Y)), (∀ x, g x ∈ t) ∧ g.comp ⟨e, he.continuous⟩ = f := by
  /-
    X₁ : Type u₁
    inst✝³ : TopologicalSpace X₁
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : NormalSpace X
    e : X₁ → X
    he : Topology.IsClosedEmbedding e
    Y : Type v
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X₁ Y
    t : Set Y
    hf : ∀ (x : X₁), Membership.mem t (f x)
    ht : TietzeExtension ↑t
    ⊢ Exists fun g => And (∀ (x : X), Membership.mem t (g x)) (Eq (g.comp { toFun  …
  -/
  obtain ⟨g, hg⟩ := mk _ (map_continuous f |>.codRestrict hf) |>.exists_extension he
  /-
    case intro
    X₁ : Type u₁
    inst✝³ : TopologicalSpace X₁
    X : Type u
    inst✝² : TopologicalSpace X
    inst✝¹ : NormalSpace X
    e : X₁ → X
    he : Topology.IsClosedEmbedding e
    Y : Type v
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X₁ Y
    t : Set Y
    hf : ∀ (x : X₁), Membership.mem t (f x)
    ht : TietzeExtension ↑t
    g : ContinuousMap X ↑t
    hg : Eq (g.comp { toFun := e, continuous_toFun := ⋯ }) { toFun := Set.codRestr …
    ⊢ Exists fun g => And (∀ (x : X), Membership.mem t (g x)) (Eq (g.comp { toFun  …
  -/
  exact ⟨comp ⟨Subtype.val, by continuity⟩ g, by simp, by ext x; congrm(($(hg) x : Y))⟩
  /-
    🎉 no goals
  -/


instance Pi.instTietzeExtension {ι : Type*} {Y : ι → Type v} [∀ i, TopologicalSpace (Y i)]
    [∀ i, TietzeExtension.{u} (Y i)] : TietzeExtension.{u} (∀ i, Y i) where
  exists_restrict_eq' s hs f := by
    obtain ⟨g', hg'⟩ := Classical.skolem.mp <| fun i ↦
      ContinuousMap.exists_restrict_eq hs (ContinuousMap.piEquiv _ _ |>.symm f i)
    /-
      case intro
      X₁ : Type u₁
      inst✝⁸ : TopologicalSpace X₁
      X : Type u
      inst✝⁷ : TopologicalSpace X
      inst✝⁶ : NormalSpace X
      s✝ : Set X
      e : X₁ → X
      Y✝ : Type v
      inst✝⁵ : TopologicalSpace Y✝
      inst✝⁴ : TietzeExtension Y✝
      ι : Type u_1
      Y : ι → Type v
      inst✝³ : (i : ι) → TopologicalSpace (Y i)
      inst✝² : ∀ (i : ι), TietzeExtension (Y i)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) ((i : ι) → Y i)
      g' : (x : ι) → ContinuousMap X✝ (Y x)
      hg' : ∀ (x : ι), Eq (ContinuousMap.restrict s (g' x)) ((ContinuousMap.piEquiv  …
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    exact ⟨ContinuousMap.piEquiv _ _ g', by ext x i; congrm($(hg' i) x)⟩
    /-
      🎉 no goals
    -/


instance Prod.instTietzeExtension {Y : Type v} {Z : Type w} [TopologicalSpace Y]
    [TietzeExtension.{u, v} Y] [TopologicalSpace Z] [TietzeExtension.{u, w} Z] :
    TietzeExtension.{u, max w v} (Y × Z) where
  exists_restrict_eq' s hs f := by
    /-
      X₁ : Type u₁
      inst✝¹⁰ : TopologicalSpace X₁
      X : Type u
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : NormalSpace X
      s✝ : Set X
      e : X₁ → X
      Y✝ : Type v
      inst✝⁷ : TopologicalSpace Y✝
      inst✝⁶ : TietzeExtension Y✝
      Y : Type v
      Z : Type w
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : TietzeExtension Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) (Prod Y Z)
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    obtain ⟨g₁, hg₁⟩ := (ContinuousMap.fst.comp f).exists_restrict_eq hs
    /-
      case intro
      X₁ : Type u₁
      inst✝¹⁰ : TopologicalSpace X₁
      X : Type u
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : NormalSpace X
      s✝ : Set X
      e : X₁ → X
      Y✝ : Type v
      inst✝⁷ : TopologicalSpace Y✝
      inst✝⁶ : TietzeExtension Y✝
      Y : Type v
      Z : Type w
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : TietzeExtension Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) (Prod Y Z)
      g₁ : ContinuousMap X✝ Y
      hg₁ : Eq (ContinuousMap.restrict s g₁) (ContinuousMap.fst.comp f)
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    obtain ⟨g₂, hg₂⟩ := (ContinuousMap.snd.comp f).exists_restrict_eq hs
    /-
      case intro.intro
      X₁ : Type u₁
      inst✝¹⁰ : TopologicalSpace X₁
      X : Type u
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : NormalSpace X
      s✝ : Set X
      e : X₁ → X
      Y✝ : Type v
      inst✝⁷ : TopologicalSpace Y✝
      inst✝⁶ : TietzeExtension Y✝
      Y : Type v
      Z : Type w
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : TietzeExtension Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) (Prod Y Z)
      g₁ : ContinuousMap X✝ Y
      hg₁ : Eq (ContinuousMap.restrict s g₁) (ContinuousMap.fst.comp f)
      g₂ : ContinuousMap X✝ Z
      hg₂ : Eq (ContinuousMap.restrict s g₂) (ContinuousMap.snd.comp f)
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    exact ⟨g₁.prodMk g₂, by ext1 x; congrm(($(hg₁) x), $(hg₂) x)⟩
    /-
      🎉 no goals
    -/


instance Unique.instTietzeExtension {Y : Type v} [TopologicalSpace Y]
    [Nonempty Y] [Subsingleton Y] : TietzeExtension.{u, v} Y where
                                                                         /-
                                                                           X₁ : Type u₁
                                                                           inst✝⁹ : TopologicalSpace X₁
                                                                           X : Type u
                                                                           inst✝⁸ : TopologicalSpace X
                                                                           inst✝⁷ : NormalSpace X
                                                                           s : Set X
                                                                           e : X₁ → X
                                                                           Y✝ : Type v
                                                                           inst✝⁶ : TopologicalSpace Y✝
                                                                           inst✝⁵ : TietzeExtension Y✝
                                                                           Y : Type v
                                                                           inst✝⁴ : TopologicalSpace Y
                                                                           inst✝³ : Nonempty Y
                                                                           inst✝² : Subsingleton Y
                                                                           X✝ : Type u
                                                                           inst✝¹ : TopologicalSpace X✝
                                                                           inst✝ : NormalSpace X✝
                                                                           x✝¹ : Set X✝
                                                                           x✝ : IsClosed x✝¹
                                                                           f : ContinuousMap (↑x✝¹) Y
                                                                           y : Y
                                                                           ⊢ Eq (ContinuousMap.restrict x✝¹ (ContinuousMap.const X✝ y)) f
                                                                         -/
  exists_restrict_eq' _ _ f := ‹Nonempty Y›.elim fun y ↦ ⟨.const _ y, by ext; subsingleton⟩
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- Any retract of a `TietzeExtension` space is one itself. -/
theorem TietzeExtension.of_retract {Y : Type v} {Z : Type w} [TopologicalSpace Y]
    [TopologicalSpace Z] [TietzeExtension.{u, w} Z] (ι : C(Y, Z)) (r : C(Z, Y))
    (h : r.comp ι = .id Y) : TietzeExtension.{u, v} Y where
  exists_restrict_eq' s hs f := by
    /-
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    obtain ⟨g, hg⟩ := (ι.comp f).exists_restrict_eq hs
    /-
      case intro
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      g : ContinuousMap X✝ Z
      hg : Eq (ContinuousMap.restrict s g) (ι.comp f)
      ⊢ Exists fun g => Eq (ContinuousMap.restrict s g) f
    -/
    use r.comp g
    /-
      case h
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      g : ContinuousMap X✝ Z
      hg : Eq (ContinuousMap.restrict s g) (ι.comp f)
      ⊢ Eq (ContinuousMap.restrict s (r.comp g)) f
    -/
    ext1 x
    /-
      case h.h
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      g : ContinuousMap X✝ Z
      hg : Eq (ContinuousMap.restrict s g) (ι.comp f)
      x : ↑s
      ⊢ Eq ((ContinuousMap.restrict s (r.comp g)) x) (f x)
    -/
    have := congr(r.comp $(hg))
    /-
      case h.h
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      g : ContinuousMap X✝ Z
      hg : Eq (ContinuousMap.restrict s g) (ι.comp f)
      x : ↑s
      this : Eq (r.comp (ContinuousMap.restrict s g)) (r.comp (ι.comp f))
      ⊢ Eq ((ContinuousMap.restrict s (r.comp g)) x) (f x)
    -/
    rw [← r.comp_assoc ι, h, f.id_comp] at this
    /-
      case h.h
      Y : Type v
      Z : Type w
      inst✝⁴ : TopologicalSpace Y
      inst✝³ : TopologicalSpace Z
      inst✝² : TietzeExtension Z
      ι : ContinuousMap Y Z
      r : ContinuousMap Z Y
      h : Eq (r.comp ι) (ContinuousMap.id Y)
      X✝ : Type u
      inst✝¹ : TopologicalSpace X✝
      inst✝ : NormalSpace X✝
      s : Set X✝
      hs : IsClosed s
      f : ContinuousMap (↑s) Y
      g : ContinuousMap X✝ Z
      hg : Eq (ContinuousMap.restrict s g) (ι.comp f)
      x : ↑s
      this : Eq (r.comp (ContinuousMap.restrict s g)) f
      ⊢ Eq ((ContinuousMap.restrict s (r.comp g)) x) (f x)
    -/
    congrm($this x)
    /-
      🎉 no goals
    -/


/-- Any homeomorphism from a `TietzeExtension` space is one itself. -/
theorem TietzeExtension.of_homeo {Y : Type v} {Z : Type w} [TopologicalSpace Y]
    [TopologicalSpace Z] [TietzeExtension.{u, w} Z] (e : Y ≃ₜ Z) :
    TietzeExtension.{u, v} Y :=
                                                     /-
                                                       Y : Type v
                                                       Z : Type w
                                                       inst✝² : TopologicalSpace Y
                                                       inst✝¹ : TopologicalSpace Z
                                                       inst✝ : TietzeExtension Z
                                                       e : Homeomorph Y Z
                                                       ⊢ Eq ((↑e.symm).comp ↑e) (ContinuousMap.id Y)
                                                     -/
  .of_retract (e : C(Y, Z)) (e.symm : C(Z, Y)) <| by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- One step in the proof of the Tietze extension theorem. If `e : C(X, Y)` is a closed embedding
of a topological space into a normal topological space and `f : X →ᵇ ℝ` is a bounded continuous
function, then there exists a bounded continuous function `g : Y →ᵇ ℝ` of the norm `‖g‖ ≤ ‖f‖ / 3`
such that the distance between `g ∘ e` and `f` is at most `(2 / 3) * ‖f‖`. -/
theorem tietze_extension_step (f : X →ᵇ ℝ) (e : C(X, Y)) (he : IsClosedEmbedding e) :
    ∃ g : Y →ᵇ ℝ, ‖g‖ ≤ ‖f‖ / 3 ∧ dist (g.compContinuous e) f ≤ 2 / 3 * ‖f‖ := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  have h3 : (0 : ℝ) < 3 := by norm_num1
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  have h23 : 0 < (2 / 3 : ℝ) := by norm_num1
  -- In the trivial case `f = 0`, we take `g = 0`
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    h23 : LT.lt 0 (2 / 3)
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  rcases eq_or_ne f 0 with (rfl | hf)
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm 0) 3)) (LE.le …
    -/
  · use 0
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      ⊢ And (LE.le (Norm.norm 0) (HDiv.hDiv (Norm.norm 0) 3)) (LE.le (Dist.dist (Bou …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    h23 : LT.lt 0 (2 / 3)
    hf : Ne f 0
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  replace hf : 0 < ‖f‖ := norm_pos_iff.2 hf
  /- Otherwise, the closed sets `e '' (f ⁻¹' (Iic (-‖f‖ / 3)))` and `e '' (f ⁻¹' (Ici (‖f‖ / 3)))`
    are disjoint, hence by Urysohn's lemma there exists a function `g` that is equal to `-‖f‖ / 3`
    on the former set and is equal to `‖f‖ / 3` on the latter set. This function `g` satisfies the
    assertions of the lemma. -/
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    h23 : LT.lt 0 (2 / 3)
    hf : LT.lt 0 (Norm.norm f)
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  have hf3 : -‖f‖ / 3 < ‖f‖ / 3 := (div_lt_div_iff_of_pos_right h3).2 (Left.neg_lt_self hf)
  have hc₁ : IsClosed (e '' (f ⁻¹' Iic (-‖f‖ / 3))) :=
    he.isClosedMap _ (isClosed_Iic.preimage f.continuous)
  have hc₂ : IsClosed (e '' (f ⁻¹' Ici (‖f‖ / 3))) :=
    he.isClosedMap _ (isClosed_Ici.preimage f.continuous)
  have hd : Disjoint (e '' (f ⁻¹' Iic (-‖f‖ / 3))) (e '' (f ⁻¹' Ici (‖f‖ / 3))) := by
    refine disjoint_image_of_injective he.injective (Disjoint.preimage _ ?_)
    rwa [Iic_disjoint_Ici, not_le]
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    h23 : LT.lt 0 (2 / 3)
    hf : LT.lt 0 (Norm.norm f)
    hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
    hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
    hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
    hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  rcases exists_bounded_mem_Icc_of_closed_of_le hc₁ hc₂ hd hf3.le with ⟨g, hg₁, hg₂, hgf⟩
  /-
    case inr.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    h3 : LT.lt 0 3
    h23 : LT.lt 0 (2 / 3)
    hf : LT.lt 0 (Norm.norm f)
    hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
    hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
    hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
    hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
    g : BoundedContinuousFunction Y Real
    hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
    hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
    hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
    ⊢ Exists fun g => And (LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)) (LE.le …
  -/
  refine ⟨g, ?_, ?_⟩
    /-
      case inr.intro.intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      hf : LT.lt 0 (Norm.norm f)
      hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
      hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
      hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
      hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
      g : BoundedContinuousFunction Y Real
      hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
      hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
      hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
      ⊢ LE.le (Norm.norm g) (HDiv.hDiv (Norm.norm f) 3)
    -/
  · refine (norm_le <| div_nonneg hf.le h3.le).mpr fun y => ?_
    /-
      case inr.intro.intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      hf : LT.lt 0 (Norm.norm f)
      hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
      hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
      hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
      hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
      g : BoundedContinuousFunction Y Real
      hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
      hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
      hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
      y : Y
      ⊢ LE.le (Norm.norm (g y)) (HDiv.hDiv (Norm.norm f) 3)
    -/
    simpa [abs_le, neg_div] using hgf y
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      hf : LT.lt 0 (Norm.norm f)
      hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
      hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
      hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
      hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
      g : BoundedContinuousFunction Y Real
      hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
      hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
      hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
      ⊢ LE.le (Dist.dist (g.compContinuous e) f) (HMul.hMul (2 / 3) (Norm.norm f))
    -/
  · refine (dist_le <| mul_nonneg h23.le hf.le).mpr fun x => ?_
    have hfx : -‖f‖ ≤ f x ∧ f x ≤ ‖f‖ := by
      simpa only [Real.norm_eq_abs, abs_le] using f.norm_coe_le_norm x
    /-
      case inr.intro.intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      h3 : LT.lt 0 3
      h23 : LT.lt 0 (2 / 3)
      hf : LT.lt 0 (Norm.norm f)
      hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
      hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
      hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
      hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
      g : BoundedContinuousFunction Y Real
      hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
      hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
      hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
      x : X
      hfx : And (LE.le (Neg.neg (Norm.norm f)) (f x)) (LE.le (f x) (Norm.norm f))
      ⊢ LE.le (Dist.dist ((g.compContinuous e) x) (f x)) (HMul.hMul (2 / 3) (Norm.no …
    -/
    rcases le_total (f x) (-‖f‖ / 3) with hle₁ | hle₁
    · calc
        |g (e x) - f x| = -‖f‖ / 3 - f x := by
          rw [hg₁ (mem_image_of_mem _ hle₁), Function.const_apply,
            abs_of_nonneg (sub_nonneg.2 hle₁)]
        _ ≤ 2 / 3 * ‖f‖ := by linarith
      /-
        case inr.intro.intro.intro.refine_2.inr
        X : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        inst✝ : NormalSpace Y
        f : BoundedContinuousFunction X Real
        e : ContinuousMap X Y
        he : Topology.IsClosedEmbedding ⇑e
        h3 : LT.lt 0 3
        h23 : LT.lt 0 (2 / 3)
        hf : LT.lt 0 (Norm.norm f)
        hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
        hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
        hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
        hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
        g : BoundedContinuousFunction Y Real
        hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
        hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
        hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
        x : X
        hfx : And (LE.le (Neg.neg (Norm.norm f)) (f x)) (LE.le (f x) (Norm.norm f))
        hle₁ : LE.le (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (f x)
        ⊢ LE.le (Dist.dist ((g.compContinuous e) x) (f x)) (HMul.hMul (2 / 3) (Norm.no …
      -/
    · rcases le_total (f x) (‖f‖ / 3) with hle₂ | hle₂
        /-
          case inr.intro.intro.intro.refine_2.inr.inl
          X : Type u_1
          Y : Type u_2
          inst✝² : TopologicalSpace X
          inst✝¹ : TopologicalSpace Y
          inst✝ : NormalSpace Y
          f : BoundedContinuousFunction X Real
          e : ContinuousMap X Y
          he : Topology.IsClosedEmbedding ⇑e
          h3 : LT.lt 0 3
          h23 : LT.lt 0 (2 / 3)
          hf : LT.lt 0 (Norm.norm f)
          hf3 : LT.lt (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (HDiv.hDiv (Norm.norm f) 3)
          hc₁ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg …
          hc₂ : IsClosed (Set.image (⇑e) (Set.preimage (⇑f) (Set.Ici (HDiv.hDiv (Norm.no …
          hd : Disjoint (Set.image (⇑e) (Set.preimage (⇑f) (Set.Iic (HDiv.hDiv (Neg.neg  …
          g : BoundedContinuousFunction Y Real
          hg₁ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Neg.neg (Norm.norm f)) 3)) ( …
          hg₂ : Set.EqOn (⇑g) (Function.const Y (HDiv.hDiv (Norm.norm f) 3)) (Set.image  …
          hgf : ∀ (x : Y), Membership.mem (Set.Icc (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) …
          x : X
          hfx : And (LE.le (Neg.neg (Norm.norm f)) (f x)) (LE.le (f x) (Norm.norm f))
          hle₁ : LE.le (HDiv.hDiv (Neg.neg (Norm.norm f)) 3) (f x)
          hle₂ : LE.le (f x) (HDiv.hDiv (Norm.norm f) 3)
          ⊢ LE.le (Dist.dist ((g.compContinuous e) x) (f x)) (HMul.hMul (2 / 3) (Norm.no …
        -/
      · simp only [neg_div] at *
        calc
          dist (g (e x)) (f x) ≤ |g (e x)| + |f x| := dist_le_norm_add_norm _ _
          _ ≤ ‖f‖ / 3 + ‖f‖ / 3 := (add_le_add (abs_le.2 <| hgf _) (abs_le.2 ⟨hle₁, hle₂⟩))
          _ = 2 / 3 * ‖f‖ := by linarith
      · calc
          |g (e x) - f x| = f x - ‖f‖ / 3 := by
            rw [hg₂ (mem_image_of_mem _ hle₂), abs_sub_comm, Function.const_apply,
              abs_of_nonneg (sub_nonneg.2 hle₂)]
          _ ≤ 2 / 3 * ‖f‖ := by linarith


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version with a closed
embedding and bundled composition. If `e : C(X, Y)` is a closed embedding of a topological space
into a normal topological space and `f : X →ᵇ ℝ` is a bounded continuous function, then there exists
a bounded continuous function `g : Y →ᵇ ℝ` of the same norm such that `g ∘ e = f`. -/
theorem exists_extension_norm_eq_of_isClosedEmbedding' (f : X →ᵇ ℝ) (e : C(X, Y))
    (he : IsClosedEmbedding e) : ∃ g : Y →ᵇ ℝ, ‖g‖ = ‖f‖ ∧ g.compContinuous e = f := by
  /- For the proof, we iterate `tietze_extension_step`. Each time we apply it to the difference
    between the previous approximation and `f`. -/
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.compContinuous e …
  -/
  choose F hF_norm hF_dist using fun f : X →ᵇ ℝ => tietze_extension_step f e he
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
    hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
    hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.compContinuous e …
  -/
  set g : ℕ → Y →ᵇ ℝ := fun n => (fun g => g + F (f - g.compContinuous e))^[n] 0
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
    hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
    hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
    g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.compContinuous e …
  -/
  have g0 : g 0 = 0 := rfl
  have g_succ : ∀ n, g (n + 1) = g n + F (f - (g n).compContinuous e) := fun n =>
    Function.iterate_succ_apply' _ _ _
  have hgf : ∀ n, dist ((g n).compContinuous e) f ≤ (2 / 3) ^ n * ‖f‖ := by
    intro n
    induction n with
    | zero => simp [g0]
    | succ n ihn =>
      rw [g_succ n, add_compContinuous, ← dist_sub_right, add_sub_cancel_left, pow_succ', mul_assoc]
      refine (hF_dist _).trans (mul_le_mul_of_nonneg_left ?_ (by norm_num1))
      rwa [← dist_eq_norm']
  have hg_dist : ∀ n, dist (g n) (g (n + 1)) ≤ 1 / 3 * ‖f‖ * (2 / 3) ^ n := by
    intro n
    calc
      dist (g n) (g (n + 1)) = ‖F (f - (g n).compContinuous e)‖ := by
        rw [g_succ, dist_eq_norm', add_sub_cancel_left]
      _ ≤ ‖f - (g n).compContinuous e‖ / 3 := hF_norm _
      _ = 1 / 3 * dist ((g n).compContinuous e) f := by rw [dist_eq_norm', one_div, div_eq_inv_mul]
      _ ≤ 1 / 3 * ((2 / 3) ^ n * ‖f‖) := mul_le_mul_of_nonneg_left (hgf n) (by norm_num1)
      _ = 1 / 3 * ‖f‖ * (2 / 3) ^ n := by ac_rfl
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
    hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
    hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
    g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
    g0 : Eq (g 0) 0
    g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
    hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
    hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.compContinuous e …
  -/
  have hg_cau : CauchySeq g := cauchySeq_of_le_geometric _ _ (by norm_num1) hg_dist
  have :
    Tendsto (fun n => (g n).compContinuous e) atTop
      (𝓝 <| (limUnder atTop g).compContinuous e) :=
    ((continuous_compContinuous e).tendsto _).comp hg_cau.tendsto_limUnder
  have hge : (limUnder atTop g).compContinuous e = f := by
    refine tendsto_nhds_unique this (tendsto_iff_dist_tendsto_zero.2 ?_)
    refine squeeze_zero (fun _ => dist_nonneg) hgf ?_
    rw [← zero_mul ‖f‖]
    refine (tendsto_pow_atTop_nhds_zero_of_lt_one ?_ ?_).mul tendsto_const_nhds <;> norm_num1
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : ContinuousMap X Y
    he : Topology.IsClosedEmbedding ⇑e
    F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
    hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
    hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
    g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
    g0 : Eq (g 0) 0
    g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
    hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
    hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
    hg_cau : CauchySeq g
    this : Filter.Tendsto (fun n => (g n).compContinuous e) Filter.atTop (nhds ((l …
    hge : Eq ((limUnder Filter.atTop g).compContinuous e) f
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (g.compContinuous e …
  -/
  refine ⟨limUnder atTop g, le_antisymm ?_ ?_, hge⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
      hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
      hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
      g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
      g0 : Eq (g 0) 0
      g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
      hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
      hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
      hg_cau : CauchySeq g
      this : Filter.Tendsto (fun n => (g n).compContinuous e) Filter.atTop (nhds ((l …
      hge : Eq ((limUnder Filter.atTop g).compContinuous e) f
      ⊢ LE.le (Norm.norm (limUnder Filter.atTop g)) (Norm.norm f)
    -/
  · rw [← dist_zero_left, ← g0]
    refine
      (dist_le_of_le_geometric_of_tendsto₀ _ _ (by norm_num1)
        hg_dist hg_cau.tendsto_limUnder).trans_eq ?_
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
      hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
      hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
      g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
      g0 : Eq (g 0) 0
      g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
      hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
      hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
      hg_cau : CauchySeq g
      this : Filter.Tendsto (fun n => (g n).compContinuous e) Filter.atTop (nhds ((l …
      hge : Eq ((limUnder Filter.atTop g).compContinuous e) f
      ⊢ Eq (HDiv.hDiv (HMul.hMul (1 / 3) (Norm.norm f)) (HSub.hSub 1 (2 / 3))) (Norm …
    -/
    field_simp [show (3 - 2 : ℝ) = 1 by norm_num1]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
      hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
      hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
      g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
      g0 : Eq (g 0) 0
      g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
      hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
      hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
      hg_cau : CauchySeq g
      this : Filter.Tendsto (fun n => (g n).compContinuous e) Filter.atTop (nhds ((l …
      hge : Eq ((limUnder Filter.atTop g).compContinuous e) f
      ⊢ LE.le (Norm.norm f) (Norm.norm (limUnder Filter.atTop g))
    -/
  · rw [← hge]
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      e : ContinuousMap X Y
      he : Topology.IsClosedEmbedding ⇑e
      F : BoundedContinuousFunction X Real → BoundedContinuousFunction Y Real
      hF_norm : ∀ (f : BoundedContinuousFunction X Real), LE.le (Norm.norm (F f)) (H …
      hF_dist : ∀ (f : BoundedContinuousFunction X Real), LE.le (Dist.dist ((F f).co …
      g : Nat → BoundedContinuousFunction Y Real := fun n => Nat.iterate (fun g => H …
      g0 : Eq (g 0) 0
      g_succ : ∀ (n : Nat), Eq (g (HAdd.hAdd n 1)) (HAdd.hAdd (g n) (F (HSub.hSub f  …
      hgf : ∀ (n : Nat), LE.le (Dist.dist ((g n).compContinuous e) f) (HMul.hMul (HP …
      hg_dist : ∀ (n : Nat), LE.le (Dist.dist (g n) (g (HAdd.hAdd n 1))) (HMul.hMul  …
      hg_cau : CauchySeq g
      this : Filter.Tendsto (fun n => (g n).compContinuous e) Filter.atTop (nhds ((l …
      hge : Eq ((limUnder Filter.atTop g).compContinuous e) f
      ⊢ LE.le (Norm.norm ((limUnder Filter.atTop g).compContinuous e)) (Norm.norm (l …
    -/
    exact norm_compContinuous_le _ _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_norm_eq_of_closedEmbedding' := exists_extension_norm_eq_of_isClosedEmbedding'


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version with a closed
embedding and unbundled composition. If `e : C(X, Y)` is a closed embedding of a topological space
into a normal topological space and `f : X →ᵇ ℝ` is a bounded continuous function, then there exists
a bounded continuous function `g : Y →ᵇ ℝ` of the same norm such that `g ∘ e = f`. -/
theorem exists_extension_norm_eq_of_isClosedEmbedding (f : X →ᵇ ℝ) {e : X → Y}
    (he : IsClosedEmbedding e) : ∃ g : Y →ᵇ ℝ, ‖g‖ = ‖f‖ ∧ g ∘ e = f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    ⊢ Exists fun g => And (Eq (Norm.norm g) (Norm.norm f)) (Eq (Function.comp (⇑g) …
  -/
  rcases exists_extension_norm_eq_of_isClosedEmbedding' f ⟨e, he.continuous⟩ he with ⟨g, hg, rfl⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    e : X → Y
    he : Topology.IsClosedEmbedding e
    g : BoundedContinuousFunction Y Real
    hg : Eq (Norm.norm g) (Norm.norm (g.compContinuous { toFun := e, continuous_to …
    ⊢ Exists fun g_1 => And (Eq (Norm.norm g_1) (Norm.norm (g.compContinuous { toF …
  -/
  exact ⟨g, hg, rfl⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_norm_eq_of_closedEmbedding := exists_extension_norm_eq_of_isClosedEmbedding


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version for a closed
set. If `f` is a bounded continuous real-valued function defined on a closed set in a normal
topological space, then it can be extended to a bounded continuous function of the same norm defined
on the whole space. -/
theorem exists_norm_eq_restrict_eq_of_closed {s : Set Y} (f : s →ᵇ ℝ) (hs : IsClosed s) :
    ∃ g : Y →ᵇ ℝ, ‖g‖ = ‖f‖ ∧ g.restrict s = f :=
  exists_extension_norm_eq_of_isClosedEmbedding' f ((ContinuousMap.id _).restrict s)
    hs.isClosedEmbedding_subtypeVal


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version for a closed
embedding and a bounded continuous function that takes values in a non-trivial closed interval.
See also `exists_extension_forall_mem_of_isClosedEmbedding` for a more general statement that works
for any interval (finite or infinite, open or closed).

If `e : X → Y` is a closed embedding and `f : X →ᵇ ℝ` is a bounded continuous function such that
`f x ∈ [a, b]` for all `x`, where `a ≤ b`, then there exists a bounded continuous function
`g : Y →ᵇ ℝ` such that `g y ∈ [a, b]` for all `y` and `g ∘ e = f`. -/
theorem exists_extension_forall_mem_Icc_of_isClosedEmbedding (f : X →ᵇ ℝ) {a b : ℝ} {e : X → Y}
    (hf : ∀ x, f x ∈ Icc a b) (hle : a ≤ b) (he : IsClosedEmbedding e) :
    ∃ g : Y →ᵇ ℝ, (∀ y, g y ∈ Icc a b) ∧ g ∘ e = f := by
  rcases exists_extension_norm_eq_of_isClosedEmbedding (f - const X ((a + b) / 2)) he with
    ⟨g, hgf, hge⟩
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    a b : Real
    e : X → Y
    hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    he : Topology.IsClosedEmbedding e
    g : BoundedContinuousFunction Y Real
    hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
    hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem (Set.Icc a b) (g y)) (Eq (Fun …
  -/
  refine ⟨const Y ((a + b) / 2) + g, fun y => ?_, ?_⟩
  · suffices ‖f - const X ((a + b) / 2)‖ ≤ (b - a) / 2 by
      simpa [Real.Icc_eq_closedBall, add_mem_closedBall_iff_norm] using
        (norm_coe_le_norm g y).trans (hgf.trans_le this)
    /-
      case intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      a b : Real
      e : X → Y
      hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      he : Topology.IsClosedEmbedding e
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
      hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
      y : Y
      ⊢ LE.le (Norm.norm (HSub.hSub f (BoundedContinuousFunction.const X (HDiv.hDiv  …
    -/
    refine (norm_le <| div_nonneg (sub_nonneg.2 hle) zero_le_two).2 fun x => ?_
    /-
      case intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      a b : Real
      e : X → Y
      hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      he : Topology.IsClosedEmbedding e
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
      hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
      y : Y
      x : X
      ⊢ LE.le (Norm.norm ((HSub.hSub f (BoundedContinuousFunction.const X (HDiv.hDiv …
    -/
    simpa only [Real.Icc_eq_closedBall] using hf x
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      a b : Real
      e : X → Y
      hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      he : Topology.IsClosedEmbedding e
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
      hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
      ⊢ Eq (Function.comp (⇑(HAdd.hAdd (BoundedContinuousFunction.const Y (HDiv.hDiv …
    -/
  · ext x
    /-
      case intro.intro.refine_2.h
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      a b : Real
      e : X → Y
      hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      he : Topology.IsClosedEmbedding e
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
      hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
      x : X
      ⊢ Eq (Function.comp (⇑(HAdd.hAdd (BoundedContinuousFunction.const Y (HDiv.hDiv …
    -/
    have : g (e x) = f x - (a + b) / 2 := congr_fun hge x
    /-
      case intro.intro.refine_2.h
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      a b : Real
      e : X → Y
      hf : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      he : Topology.IsClosedEmbedding e
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Norm.norm g) (Norm.norm (HSub.hSub f (BoundedContinuousFunction.cons …
      hge : Eq (Function.comp (⇑g) e) ⇑(HSub.hSub f (BoundedContinuousFunction.const …
      x : X
      this : Eq (g (e x)) (HSub.hSub (f x) (HDiv.hDiv (HAdd.hAdd a b) 2))
      ⊢ Eq (Function.comp (⇑(HAdd.hAdd (BoundedContinuousFunction.const Y (HDiv.hDiv …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_forall_mem_Icc_of_closedEmbedding :=
  exists_extension_forall_mem_Icc_of_isClosedEmbedding


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version for a closed
embedding. Let `e` be a closed embedding of a nonempty topological space `X` into a normal
topological space `Y`. Let `f` be a bounded continuous real-valued function on `X`. Then there
exists a bounded continuous function `g : Y →ᵇ ℝ` such that `g ∘ e = f` and each value `g y` belongs
to a closed interval `[f x₁, f x₂]` for some `x₁` and `x₂`. -/
theorem exists_extension_forall_exists_le_ge_of_isClosedEmbedding [Nonempty X] (f : X →ᵇ ℝ)
    {e : X → Y} (he : IsClosedEmbedding e) :
    ∃ g : Y →ᵇ ℝ, (∀ y, ∃ x₁ x₂, g y ∈ Icc (f x₁) (f x₂)) ∧ g ∘ e = f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  inhabit X
  -- Put `a = ⨅ x, f x` and `b = ⨆ x, f x`
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  obtain ⟨a, ha⟩ : ∃ a, IsGLB (range f) a := ⟨_, isGLB_ciInf f.isBounded_range.bddBelow⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  obtain ⟨b, hb⟩ : ∃ b, IsLUB (range f) b := ⟨_, isLUB_ciSup f.isBounded_range.bddAbove⟩
  -- Then `f x ∈ [a, b]` for all `x`
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  have hmem : ∀ x, f x ∈ Icc a b := fun x => ⟨ha.1 ⟨x, rfl⟩, hb.1 ⟨x, rfl⟩⟩
  -- Rule out the trivial case `a = b`
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  have hle : a ≤ b := (hmem default).1.trans (hmem default).2
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  rcases hle.eq_or_lt with (rfl | hlt)
    /-
      case intro.intro.inl
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      hb : IsLUB (Set.range ⇑f) a
      hmem : ∀ (x : X), Membership.mem (Set.Icc a a) (f x)
      hle : LE.le a a
      ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
    -/
  · have : ∀ x, f x = a := by simpa using hmem
    /-
      case intro.intro.inl
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      hb : IsLUB (Set.range ⇑f) a
      hmem : ∀ (x : X), Membership.mem (Set.Icc a a) (f x)
      hle : LE.le a a
      this : ∀ (x : X), Eq (f x) a
      ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
    -/
    use const Y a
    /-
      case h
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      hb : IsLUB (Set.range ⇑f) a
      hmem : ∀ (x : X), Membership.mem (Set.Icc a a) (f x)
      hle : LE.le a a
      this : ∀ (x : X), Eq (f x) a
      ⊢ And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f …
    -/
    simp [this, funext_iff]
    /-
      🎉 no goals
    -/
  -- Put `c = (a + b) / 2`. Then `a < c < b` and `c - a = b - c`.
  /-
    case intro.intro.inr
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  set c := (a + b) / 2
  /-
    case intro.intro.inr
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  have hac : a < c := left_lt_add_div_two.2 hlt
  /-
    case intro.intro.inr
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  have hcb : c < b := add_div_two_lt_right.2 hlt
  have hsub : c - a = b - c := by
    field_simp [c]
    ring
  /- Due to `exists_extension_forall_mem_Icc_of_isClosedEmbedding`, there exists an extension `g`
    such that `g y ∈ [a, b]` for all `y`. However, if `a` and/or `b` do not belong to the range of
    `f`, then we need to ensure that these points do not belong to the range of `g`. This is done
    in two almost identical steps. First we deal with the case `∀ x, f x ≠ a`. -/
  obtain ⟨g, hg_mem, hgf⟩ : ∃ g : Y →ᵇ ℝ, (∀ y, ∃ x, g y ∈ Icc (f x) b) ∧ g ∘ e = f := by
    rcases exists_extension_forall_mem_Icc_of_isClosedEmbedding f hmem hle he with ⟨g, hg_mem, hgf⟩
    -- If `a ∈ range f`, then we are done.
    rcases em (∃ x, f x = a) with (⟨x, rfl⟩ | ha')
    · exact ⟨g, fun y => ⟨x, hg_mem _⟩, hgf⟩
    /- Otherwise, `g ⁻¹' {a}` is disjoint with `range e ∪ g ⁻¹' (Ici c)`, hence there exists a
        function `dg : Y → ℝ` such that `dg ∘ e = 0`, `dg y = 0` whenever `c ≤ g y`, `dg y = c - a`
        whenever `g y = a`, and `0 ≤ dg y ≤ c - a` for all `y`. -/
    have hd : Disjoint (range e ∪ g ⁻¹' Ici c) (g ⁻¹' {a}) := by
      refine disjoint_union_left.2 ⟨?_, Disjoint.preimage _ ?_⟩
      · rw [Set.disjoint_left]
        rintro _ ⟨x, rfl⟩ (rfl : g (e x) = a)
        exact ha' ⟨x, (congr_fun hgf x).symm⟩
      · exact Set.disjoint_singleton_right.2 hac.not_le
    rcases exists_bounded_mem_Icc_of_closed_of_le
        (he.isClosed_range.union <| isClosed_Ici.preimage g.continuous)
        (isClosed_singleton.preimage g.continuous) hd (sub_nonneg.2 hac.le) with
      ⟨dg, dg0, dga, dgmem⟩
    replace hgf : ∀ x, (g + dg) (e x) = f x := by
      intro x
      simp [dg0 (Or.inl <| mem_range_self _), ← hgf]
    refine ⟨g + dg, fun y => ?_, funext hgf⟩
    have hay : a < (g + dg) y := by
      rcases (hg_mem y).1.eq_or_lt with (rfl | hlt)
      · refine (lt_add_iff_pos_right _).2 ?_
        calc
          0 < c - g y := sub_pos.2 hac
          _ = dg y := (dga rfl).symm
      · exact hlt.trans_le (le_add_of_nonneg_right (dgmem y).1)
    rcases ha.exists_between hay with ⟨_, ⟨x, rfl⟩, _, hxy⟩
    refine ⟨x, hxy.le, ?_⟩
    rcases le_total c (g y) with hc | hc
    · simp [dg0 (Or.inr hc), (hg_mem y).2]
    · calc
        g y + dg y ≤ c + (c - a) := add_le_add hc (dgmem _).2
        _ = b := by rw [hsub, add_sub_cancel]
  /- Now we deal with the case `∀ x, f x ≠ b`. The proof is the same as in the first case, with
    minor modifications that make it hard to deduplicate code. -/
  /-
    case intro.intro.inr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    hcb : LT.lt c b
    hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
    g : BoundedContinuousFunction Y Real
    hg_mem : ∀ (y : Y), Exists fun x => Membership.mem (Set.Icc (f x) b) (g y)
    hgf : Eq (Function.comp (⇑g) e) ⇑f
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  choose xl hxl hgb using hg_mem
  /-
    case intro.intro.inr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    hcb : LT.lt c b
    hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
    g : BoundedContinuousFunction Y Real
    hgf : Eq (Function.comp (⇑g) e) ⇑f
    xl : Y → X
    hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
    hgb : ∀ (y : Y), LE.le (g y) b
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  rcases em (∃ x, f x = b) with (⟨x, rfl⟩ | hb')
    /-
      case intro.intro.inr.intro.intro.inl.intro
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      g : BoundedContinuousFunction Y Real
      hgf : Eq (Function.comp (⇑g) e) ⇑f
      xl : Y → X
      hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
      x : X
      hb : IsLUB (Set.range ⇑f) (f x)
      hmem : ∀ (x_1 : X), Membership.mem (Set.Icc a (f x)) (f x_1)
      hle : LE.le a (f x)
      hlt : LT.lt a (f x)
      c : Real := HDiv.hDiv (HAdd.hAdd a (f x)) 2
      hac : LT.lt a c
      hcb : LT.lt c (f x)
      hsub : Eq (HSub.hSub c a) (HSub.hSub (f x) c)
      hgb : ∀ (y : Y), LE.le (g y) (f x)
      ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
    -/
  · exact ⟨g, fun y => ⟨xl y, x, hxl y, hgb y⟩, hgf⟩
    /-
      🎉 no goals
    -/
  have hd : Disjoint (range e ∪ g ⁻¹' Iic c) (g ⁻¹' {b}) := by
    refine disjoint_union_left.2 ⟨?_, Disjoint.preimage _ ?_⟩
    · rw [Set.disjoint_left]
      rintro _ ⟨x, rfl⟩ (rfl : g (e x) = b)
      exact hb' ⟨x, (congr_fun hgf x).symm⟩
    · exact Set.disjoint_singleton_right.2 hcb.not_le
  rcases exists_bounded_mem_Icc_of_closed_of_le
      (he.isClosed_range.union <| isClosed_Iic.preimage g.continuous)
      (isClosed_singleton.preimage g.continuous) hd (sub_nonneg.2 hcb.le) with
    ⟨dg, dg0, dgb, dgmem⟩
  replace hgf : ∀ x, (g - dg) (e x) = f x := by
    intro x
    simp [dg0 (Or.inl <| mem_range_self _), ← hgf]
  /-
    case intro.intro.inr.intro.intro.inr.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    hcb : LT.lt c b
    hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
    g : BoundedContinuousFunction Y Real
    xl : Y → X
    hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
    hgb : ∀ (y : Y), LE.le (g y) b
    hb' : Not (Exists fun x => Eq (f x) b)
    hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
    dg : BoundedContinuousFunction Y Real
    dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
    dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
    dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
    hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
    ⊢ Exists fun g => And (∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership …
  -/
  refine ⟨g - dg, fun y => ?_, funext hgf⟩
  have hyb : (g - dg) y < b := by
    rcases (hgb y).eq_or_lt with (rfl | hlt)
    · refine (sub_lt_self_iff _).2 ?_
      calc
        0 < g y - c := sub_pos.2 hcb
        _ = dg y := (dgb rfl).symm
    · exact ((sub_le_self_iff _).2 (dgmem _).1).trans_lt hlt
  /-
    case intro.intro.inr.intro.intro.inr.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    hcb : LT.lt c b
    hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
    g : BoundedContinuousFunction Y Real
    xl : Y → X
    hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
    hgb : ∀ (y : Y), LE.le (g y) b
    hb' : Not (Exists fun x => Eq (f x) b)
    hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
    dg : BoundedContinuousFunction Y Real
    dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
    dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
    dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
    hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
    y : Y
    hyb : LT.lt ((HSub.hSub g dg) y) b
    ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
  -/
  rcases hb.exists_between hyb with ⟨_, ⟨xu, rfl⟩, hyxu, _⟩
  /-
    case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    inst✝ : Nonempty X
    f : BoundedContinuousFunction X Real
    e : X → Y
    he : Topology.IsClosedEmbedding e
    inhabited_h : Inhabited X
    a : Real
    ha : IsGLB (Set.range ⇑f) a
    b : Real
    hb : IsLUB (Set.range ⇑f) b
    hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
    hle : LE.le a b
    hlt : LT.lt a b
    c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
    hac : LT.lt a c
    hcb : LT.lt c b
    hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
    g : BoundedContinuousFunction Y Real
    xl : Y → X
    hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
    hgb : ∀ (y : Y), LE.le (g y) b
    hb' : Not (Exists fun x => Eq (f x) b)
    hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
    dg : BoundedContinuousFunction Y Real
    dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
    dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
    dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
    hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
    y : Y
    hyb : LT.lt ((HSub.hSub g dg) y) b
    xu : X
    hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
    right✝ : LE.le (f xu) b
    ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
  -/
  cases' lt_or_le c (g y) with hc hc
    /-
      case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      b : Real
      hb : IsLUB (Set.range ⇑f) b
      hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      hlt : LT.lt a b
      c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
      hac : LT.lt a c
      hcb : LT.lt c b
      hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
      g : BoundedContinuousFunction Y Real
      xl : Y → X
      hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
      hgb : ∀ (y : Y), LE.le (g y) b
      hb' : Not (Exists fun x => Eq (f x) b)
      hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
      dg : BoundedContinuousFunction Y Real
      dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
      dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
      dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
      hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
      y : Y
      hyb : LT.lt ((HSub.hSub g dg) y) b
      xu : X
      hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
      right✝ : LE.le (f xu) b
      hc : LT.lt c (g y)
      ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
    -/
  · rcases em (a ∈ range f) with (⟨x, rfl⟩ | _)
      /-
        case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
        X : Type u_1
        Y : Type u_2
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : NormalSpace Y
        inst✝ : Nonempty X
        f : BoundedContinuousFunction X Real
        e : X → Y
        he : Topology.IsClosedEmbedding e
        inhabited_h : Inhabited X
        b : Real
        hb : IsLUB (Set.range ⇑f) b
        g : BoundedContinuousFunction Y Real
        xl : Y → X
        hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
        hgb : ∀ (y : Y), LE.le (g y) b
        hb' : Not (Exists fun x => Eq (f x) b)
        dg : BoundedContinuousFunction Y Real
        hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
        y : Y
        hyb : LT.lt ((HSub.hSub g dg) y) b
        xu : X
        hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
        right✝ : LE.le (f xu) b
        x : X
        ha : IsGLB (Set.range ⇑f) (f x)
        hmem : ∀ (x_1 : X), Membership.mem (Set.Icc (f x) b) (f x_1)
        hle : LE.le (f x) b
        hlt : LT.lt (f x) b
        c : Real := HDiv.hDiv (HAdd.hAdd (f x) b) 2
        hac : LT.lt (f x) c
        hcb : LT.lt c b
        hsub : Eq (HSub.hSub c (f x)) (HSub.hSub b c)
        hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
        dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
        dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
        dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
        hc : LT.lt c (g y)
        ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
      -/
    · refine ⟨x, xu, ?_, hyxu.le⟩
      calc
        f x = c - (b - c) := by rw [← hsub, sub_sub_cancel]
        _ ≤ g y - dg y := sub_le_sub hc.le (dgmem _).2
    · have hay : a < (g - dg) y := by
        calc
          a = c - (b - c) := by rw [← hsub, sub_sub_cancel]
          _ < g y - (b - c) := sub_lt_sub_right hc _
          _ ≤ g y - dg y := sub_le_sub_left (dgmem _).2 _
      /-
        case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
        X : Type u_1
        Y : Type u_2
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : NormalSpace Y
        inst✝ : Nonempty X
        f : BoundedContinuousFunction X Real
        e : X → Y
        he : Topology.IsClosedEmbedding e
        inhabited_h : Inhabited X
        a : Real
        ha : IsGLB (Set.range ⇑f) a
        b : Real
        hb : IsLUB (Set.range ⇑f) b
        hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
        hle : LE.le a b
        hlt : LT.lt a b
        c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
        hac : LT.lt a c
        hcb : LT.lt c b
        hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
        g : BoundedContinuousFunction Y Real
        xl : Y → X
        hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
        hgb : ∀ (y : Y), LE.le (g y) b
        hb' : Not (Exists fun x => Eq (f x) b)
        hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
        dg : BoundedContinuousFunction Y Real
        dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
        dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
        dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
        hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
        y : Y
        hyb : LT.lt ((HSub.hSub g dg) y) b
        xu : X
        hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
        right✝ : LE.le (f xu) b
        hc : LT.lt c (g y)
        h✝ : Not (Membership.mem (Set.range ⇑f) a)
        hay : LT.lt a ((HSub.hSub g dg) y)
        ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
      -/
      rcases ha.exists_between hay with ⟨_, ⟨x, rfl⟩, _, hxy⟩
      /-
        case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
        X : Type u_1
        Y : Type u_2
        inst✝³ : TopologicalSpace X
        inst✝² : TopologicalSpace Y
        inst✝¹ : NormalSpace Y
        inst✝ : Nonempty X
        f : BoundedContinuousFunction X Real
        e : X → Y
        he : Topology.IsClosedEmbedding e
        inhabited_h : Inhabited X
        a : Real
        ha : IsGLB (Set.range ⇑f) a
        b : Real
        hb : IsLUB (Set.range ⇑f) b
        hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
        hle : LE.le a b
        hlt : LT.lt a b
        c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
        hac : LT.lt a c
        hcb : LT.lt c b
        hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
        g : BoundedContinuousFunction Y Real
        xl : Y → X
        hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
        hgb : ∀ (y : Y), LE.le (g y) b
        hb' : Not (Exists fun x => Eq (f x) b)
        hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
        dg : BoundedContinuousFunction Y Real
        dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
        dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
        dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
        hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
        y : Y
        hyb : LT.lt ((HSub.hSub g dg) y) b
        xu : X
        hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
        right✝ : LE.le (f xu) b
        hc : LT.lt c (g y)
        h✝ : Not (Membership.mem (Set.range ⇑f) a)
        hay : LT.lt a ((HSub.hSub g dg) y)
        x : X
        left✝ : LE.le a (f x)
        hxy : LT.lt (f x) ((HSub.hSub g dg) y)
        ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
      -/
      exact ⟨x, xu, hxy.le, hyxu.le⟩
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      b : Real
      hb : IsLUB (Set.range ⇑f) b
      hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      hlt : LT.lt a b
      c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
      hac : LT.lt a c
      hcb : LT.lt c b
      hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
      g : BoundedContinuousFunction Y Real
      xl : Y → X
      hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
      hgb : ∀ (y : Y), LE.le (g y) b
      hb' : Not (Exists fun x => Eq (f x) b)
      hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
      dg : BoundedContinuousFunction Y Real
      dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
      dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
      dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
      hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
      y : Y
      hyb : LT.lt ((HSub.hSub g dg) y) b
      xu : X
      hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
      right✝ : LE.le (f xu) b
      hc : LE.le (g y) c
      ⊢ Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x₁) (f x₂)) ((H …
    -/
  · refine ⟨xl y, xu, ?_, hyxu.le⟩
    /-
      case intro.intro.inr.intro.intro.inr.intro.intro.intro.intro.intro.intro.intro …
      X : Type u_1
      Y : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : NormalSpace Y
      inst✝ : Nonempty X
      f : BoundedContinuousFunction X Real
      e : X → Y
      he : Topology.IsClosedEmbedding e
      inhabited_h : Inhabited X
      a : Real
      ha : IsGLB (Set.range ⇑f) a
      b : Real
      hb : IsLUB (Set.range ⇑f) b
      hmem : ∀ (x : X), Membership.mem (Set.Icc a b) (f x)
      hle : LE.le a b
      hlt : LT.lt a b
      c : Real := HDiv.hDiv (HAdd.hAdd a b) 2
      hac : LT.lt a c
      hcb : LT.lt c b
      hsub : Eq (HSub.hSub c a) (HSub.hSub b c)
      g : BoundedContinuousFunction Y Real
      xl : Y → X
      hxl : ∀ (y : Y), LE.le (f (xl y)) (g y)
      hgb : ∀ (y : Y), LE.le (g y) b
      hb' : Not (Exists fun x => Eq (f x) b)
      hd : Disjoint (Union.union (Set.range e) (Set.preimage (⇑g) (Set.Iic c))) (Set …
      dg : BoundedContinuousFunction Y Real
      dg0 : Set.EqOn (⇑dg) (Function.const Y 0) (Union.union (Set.range e) (Set.prei …
      dgb : Set.EqOn (⇑dg) (Function.const Y (HSub.hSub b c)) (Set.preimage (⇑g) (Si …
      dgmem : ∀ (x : Y), Membership.mem (Set.Icc 0 (HSub.hSub b c)) (dg x)
      hgf : ∀ (x : X), Eq ((HSub.hSub g dg) (e x)) (f x)
      y : Y
      hyb : LT.lt ((HSub.hSub g dg) y) b
      xu : X
      hyxu : LT.lt ((HSub.hSub g dg) y) (f xu)
      right✝ : LE.le (f xu) b
      hc : LE.le (g y) c
      ⊢ LE.le (f (xl y)) ((HSub.hSub g dg) y)
    -/
    simp [dg0 (Or.inr hc), hxl]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_forall_exists_le_ge_of_closedEmbedding :=
  exists_extension_forall_exists_le_ge_of_isClosedEmbedding


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version for a closed
embedding. Let `e` be a closed embedding of a nonempty topological space `X` into a normal
topological space `Y`. Let `f` be a bounded continuous real-valued function on `X`. Let `t` be
a nonempty convex set of real numbers (we use `OrdConnected` instead of `Convex` to automatically
deduce this argument by typeclass search) such that `f x ∈ t` for all `x`. Then there exists
a bounded continuous real-valued function `g : Y →ᵇ ℝ` such that `g y ∈ t` for all `y` and
`g ∘ e = f`. -/
theorem exists_extension_forall_mem_of_isClosedEmbedding (f : X →ᵇ ℝ) {t : Set ℝ} {e : X → Y}
    [hs : OrdConnected t] (hf : ∀ x, f x ∈ t) (hne : t.Nonempty) (he : IsClosedEmbedding e) :
    ∃ g : Y →ᵇ ℝ, (∀ y, g y ∈ t) ∧ g ∘ e = f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  cases isEmpty_or_nonempty X
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h✝ : IsEmpty X
      ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
    -/
  · rcases hne with ⟨c, hc⟩
    /-
      case inl.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : BoundedContinuousFunction X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      he : Topology.IsClosedEmbedding e
      h✝ : IsEmpty X
      c : Real
      hc : Membership.mem t c
      ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
    -/
    exact ⟨const Y c, fun _ => hc, funext fun x => isEmptyElim x⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h✝ : Nonempty X
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  rcases exists_extension_forall_exists_le_ge_of_isClosedEmbedding f he with ⟨g, hg, hgf⟩
  /-
    case inr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h✝ : Nonempty X
    g : BoundedContinuousFunction Y Real
    hg : ∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x …
    hgf : Eq (Function.comp (⇑g) e) ⇑f
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  refine ⟨g, fun y => ?_, hgf⟩
  /-
    case inr.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h✝ : Nonempty X
    g : BoundedContinuousFunction Y Real
    hg : ∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x …
    hgf : Eq (Function.comp (⇑g) e) ⇑f
    y : Y
    ⊢ Membership.mem t (g y)
  -/
  rcases hg y with ⟨xl, xu, h⟩
  /-
    case inr.intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : BoundedContinuousFunction X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h✝ : Nonempty X
    g : BoundedContinuousFunction Y Real
    hg : ∀ (y : Y), Exists fun x₁ => Exists fun x₂ => Membership.mem (Set.Icc (f x …
    hgf : Eq (Function.comp (⇑g) e) ⇑f
    y : Y
    xl xu : X
    h : Membership.mem (Set.Icc (f xl) (f xu)) (g y)
    ⊢ Membership.mem t (g y)
  -/
  exact hs.out (hf _) (hf _) h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_forall_mem_of_closedEmbedding :=
  exists_extension_forall_mem_of_isClosedEmbedding


/-- **Tietze extension theorem** for real-valued bounded continuous maps, a version for a closed
set. Let `s` be a closed set in a normal topological space `Y`. Let `f` be a bounded continuous
real-valued function on `s`. Let `t` be a nonempty convex set of real numbers (we use
`OrdConnected` instead of `Convex` to automatically deduce this argument by typeclass search) such
that `f x ∈ t` for all `x : s`. Then there exists a bounded continuous real-valued function
`g : Y →ᵇ ℝ` such that `g y ∈ t` for all `y` and `g.restrict s = f`. -/
theorem exists_forall_mem_restrict_eq_of_closed {s : Set Y} (f : s →ᵇ ℝ) (hs : IsClosed s)
    {t : Set ℝ} [OrdConnected t] (hf : ∀ x, f x ∈ t) (hne : t.Nonempty) :
    ∃ g : Y →ᵇ ℝ, (∀ y, g y ∈ t) ∧ g.restrict s = f := by
  obtain ⟨g, hg, hgf⟩ :=
    exists_extension_forall_mem_of_isClosedEmbedding f hf hne hs.isClosedEmbedding_subtypeVal
  /-
    case intro.intro
    Y : Type u_2
    inst✝² : TopologicalSpace Y
    inst✝¹ : NormalSpace Y
    s : Set Y
    f : BoundedContinuousFunction (↑s) Real
    hs : IsClosed s
    t : Set Real
    inst✝ : t.OrdConnected
    hf : ∀ (x : ↑s), Membership.mem t (f x)
    hne : t.Nonempty
    g : BoundedContinuousFunction Y Real
    hg : ∀ (y : Y), Membership.mem t (g y)
    hgf : Eq (Function.comp (⇑g) Subtype.val) ⇑f
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (g.restrict s) f)
  -/
  exact ⟨g, hg, DFunLike.coe_injective hgf⟩
  /-
    🎉 no goals
  -/


/-- **Tietze extension theorem** for real-valued continuous maps, a version for a closed
embedding. Let `e` be a closed embedding of a nonempty topological space `X` into a normal
topological space `Y`. Let `f` be a continuous real-valued function on `X`. Let `t` be a nonempty
convex set of real numbers (we use `OrdConnected` instead of `Convex` to automatically deduce this
argument by typeclass search) such that `f x ∈ t` for all `x`. Then there exists a continuous
real-valued function `g : C(Y, ℝ)` such that `g y ∈ t` for all `y` and `g ∘ e = f`. -/
theorem exists_extension_forall_mem_of_isClosedEmbedding (f : C(X, ℝ)) {t : Set ℝ} {e : X → Y}
    [hs : OrdConnected t] (hf : ∀ x, f x ∈ t) (hne : t.Nonempty) (he : IsClosedEmbedding e) :
    ∃ g : C(Y, ℝ), (∀ y, g y ∈ t) ∧ g ∘ e = f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  have h : ℝ ≃o Ioo (-1 : ℝ) 1 := orderIsoIooNegOneOne ℝ
  let F : X →ᵇ ℝ :=
    { toFun := (↑) ∘ h ∘ f
      continuous_toFun := continuous_subtype_val.comp (h.continuous.comp f.continuous)
      map_bounded' := isBounded_range_iff.1
        ((isBounded_Ioo (-1 : ℝ) 1).subset <| range_subset_iff.2 fun x => (h (f x)).2) }
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h : OrderIso Real ↑(Set.Ioo (-1) 1)
    F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  let t' : Set ℝ := (↑) ∘ h '' t
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h : OrderIso Real ↑(Set.Ioo (-1) 1)
    F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
    t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  have ht_sub : t' ⊆ Ioo (-1 : ℝ) 1 := image_subset_iff.2 fun x _ => (h x).2
  have : OrdConnected t' := by
    constructor
    rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ z hz
    lift z to Ioo (-1 : ℝ) 1 using Icc_subset_Ioo (h x).2.1 (h y).2.2 hz
    change z ∈ Icc (h x) (h y) at hz
    rw [← h.image_Icc] at hz
    rcases hz with ⟨z, hz, rfl⟩
    exact ⟨z, hs.out hx hy hz, rfl⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h : OrderIso Real ↑(Set.Ioo (-1) 1)
    F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
    t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
    ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
    this : t'.OrdConnected
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  have hFt : ∀ x, F x ∈ t' := fun x => mem_image_of_mem _ (hf x)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h : OrderIso Real ↑(Set.Ioo (-1) 1)
    F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
    t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
    ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
    this : t'.OrdConnected
    hFt : ∀ (x : X), Membership.mem t' (F x)
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  rcases F.exists_extension_forall_mem_of_isClosedEmbedding hFt (hne.image _) he with ⟨G, hG, hGF⟩
  let g : C(Y, ℝ) :=
    ⟨h.symm ∘ codRestrict G _ fun y => ht_sub (hG y),
      h.symm.continuous.comp <| G.continuous.subtype_mk _⟩
  have hgG : ∀ {y a}, g y = a ↔ G y = h a := @fun y a =>
    h.toEquiv.symm_apply_eq.trans Subtype.ext_iff
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : NormalSpace Y
    f : ContinuousMap X Real
    t : Set Real
    e : X → Y
    hs : t.OrdConnected
    hf : ∀ (x : X), Membership.mem t (f x)
    hne : t.Nonempty
    he : Topology.IsClosedEmbedding e
    h : OrderIso Real ↑(Set.Ioo (-1) 1)
    F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
    t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
    ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
    this : t'.OrdConnected
    hFt : ∀ (x : X), Membership.mem t' (F x)
    G : BoundedContinuousFunction Y Real
    hG : ∀ (y : Y), Membership.mem t' (G y)
    hGF : Eq (Function.comp (⇑G) e) ⇑F
    g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
    hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
    ⊢ Exists fun g => And (∀ (y : Y), Membership.mem t (g y)) (Eq (Function.comp ( …
  -/
  refine ⟨g, fun y => ?_, ?_⟩
    /-
      case intro.intro.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : ContinuousMap X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h : OrderIso Real ↑(Set.Ioo (-1) 1)
      F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
      t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
      ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
      this : t'.OrdConnected
      hFt : ∀ (x : X), Membership.mem t' (F x)
      G : BoundedContinuousFunction Y Real
      hG : ∀ (y : Y), Membership.mem t' (G y)
      hGF : Eq (Function.comp (⇑G) e) ⇑F
      g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
      hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
      y : Y
      ⊢ Membership.mem t (g y)
    -/
  · rcases hG y with ⟨a, ha, hay⟩
    /-
      case intro.intro.refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : ContinuousMap X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h : OrderIso Real ↑(Set.Ioo (-1) 1)
      F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
      t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
      ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
      this : t'.OrdConnected
      hFt : ∀ (x : X), Membership.mem t' (F x)
      G : BoundedContinuousFunction Y Real
      hG : ∀ (y : Y), Membership.mem t' (G y)
      hGF : Eq (Function.comp (⇑G) e) ⇑F
      g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
      hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
      y : Y
      a : Real
      ha : Membership.mem t a
      hay : Eq (Function.comp Subtype.val (⇑h) a) (G y)
      ⊢ Membership.mem t (g y)
    -/
    convert ha
    /-
      case h.e'_5
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : ContinuousMap X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h : OrderIso Real ↑(Set.Ioo (-1) 1)
      F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
      t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
      ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
      this : t'.OrdConnected
      hFt : ∀ (x : X), Membership.mem t' (F x)
      G : BoundedContinuousFunction Y Real
      hG : ∀ (y : Y), Membership.mem t' (G y)
      hGF : Eq (Function.comp (⇑G) e) ⇑F
      g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
      hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
      y : Y
      a : Real
      ha : Membership.mem t a
      hay : Eq (Function.comp Subtype.val (⇑h) a) (G y)
      ⊢ Eq (g y) a
    -/
    exact hgG.2 hay.symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : ContinuousMap X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h : OrderIso Real ↑(Set.Ioo (-1) 1)
      F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
      t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
      ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
      this : t'.OrdConnected
      hFt : ∀ (x : X), Membership.mem t' (F x)
      G : BoundedContinuousFunction Y Real
      hG : ∀ (y : Y), Membership.mem t' (G y)
      hGF : Eq (Function.comp (⇑G) e) ⇑F
      g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
      hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
      ⊢ Eq (Function.comp (⇑g) e) ⇑f
    -/
  · ext x
    /-
      case intro.intro.refine_2.h
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      f : ContinuousMap X Real
      t : Set Real
      e : X → Y
      hs : t.OrdConnected
      hf : ∀ (x : X), Membership.mem t (f x)
      hne : t.Nonempty
      he : Topology.IsClosedEmbedding e
      h : OrderIso Real ↑(Set.Ioo (-1) 1)
      F : BoundedContinuousFunction X Real := { toFun := Function.comp Subtype.val ( …
      t' : Set Real := Set.image (Function.comp Subtype.val ⇑h) t
      ht_sub : HasSubset.Subset t' (Set.Ioo (-1) 1)
      this : t'.OrdConnected
      hFt : ∀ (x : X), Membership.mem t' (F x)
      G : BoundedContinuousFunction Y Real
      hG : ∀ (y : Y), Membership.mem t' (G y)
      hGF : Eq (Function.comp (⇑G) e) ⇑F
      g : ContinuousMap Y Real := { toFun := Function.comp (⇑h.symm) (Set.codRestric …
      hgG : ∀ {y : Y} {a : Real}, Iff (Eq (g y) a) (Eq (G y) ↑(h a))
      x : X
      ⊢ Eq (Function.comp (⇑g) e x) (f x)
    -/
    exact hgG.2 (congr_fun hGF _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias exists_extension_of_closedEmbedding := exists_extension'


/-- **Tietze extension theorem** for real-valued continuous maps, a version for a closed set. Let
`s` be a closed set in a normal topological space `Y`. Let `f` be a continuous real-valued function
on `s`. Let `t` be a nonempty convex set of real numbers (we use `OrdConnected` instead of `Convex`
to automatically deduce this argument by typeclass search) such that `f x ∈ t` for all `x : s`. Then
there exists a continuous real-valued function `g : C(Y, ℝ)` such that `g y ∈ t` for all `y` and
`g.restrict s = f`. -/
theorem exists_restrict_eq_forall_mem_of_closed {s : Set Y} (f : C(s, ℝ)) {t : Set ℝ}
    [OrdConnected t] (ht : ∀ x, f x ∈ t) (hne : t.Nonempty) (hs : IsClosed s) :
    ∃ g : C(Y, ℝ), (∀ y, g y ∈ t) ∧ g.restrict s = f :=
  let ⟨g, hgt, hgf⟩ :=
    exists_extension_forall_mem_of_isClosedEmbedding f ht hne hs.isClosedEmbedding_subtypeVal
  ⟨g, hgt, coe_injective hgf⟩


/-- **Tietze extension theorem** for real-valued continuous maps.
`ℝ` is a `TietzeExtension` space. -/
instance Real.instTietzeExtension : TietzeExtension ℝ where
  exists_restrict_eq' _s hs f :=
    f.exists_restrict_eq_forall_mem_of_closed (fun _ => mem_univ _) univ_nonempty hs |>.imp
      fun _ ↦ (And.right ·)


open NNReal in
/-- **Tietze extension theorem** for nonnegative real-valued continuous maps.
`ℝ≥0` is a `TietzeExtension` space. -/
instance NNReal.instTietzeExtension : TietzeExtension ℝ≥0 :=
                                   /-
                                     X : Type u_1
                                     Y : Type u_2
                                     inst✝² : TopologicalSpace X
                                     inst✝¹ : TopologicalSpace Y
                                     inst✝ : NormalSpace Y
                                     ⊢ Continuous NNReal.toReal
                                   -/
  .of_retract ⟨((↑) : ℝ≥0 → ℝ), by continuity⟩ ⟨Real.toNNReal, continuous_real_toNNReal⟩ <| by
                                   /-
                                     🎉 no goals
                                   -/
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : NormalSpace Y
      ⊢ Eq ({ toFun := Real.toNNReal, continuous_toFun := continuous_real_toNNReal } …
    -/
    ext; simp
         /-
           🎉 no goals
         -/

