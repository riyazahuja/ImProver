/-- A satellite configuration is a configuration of `N+1` points that shows up in the inductive
construction for the Besicovitch covering theorem. It depends on some parameter `τ ≥ 1`.

This is a family of balls (indexed by `i : Fin N.succ`, with center `c i` and radius `r i`) such
that the last ball intersects all the other balls (condition `inter`),
and given any two balls there is an order between them, ensuring that the first ball does not
contain the center of the other one, and the radius of the second ball can not be larger than
the radius of the first ball (up to a factor `τ`). This order corresponds to the order of choice
in the inductive construction: otherwise, the second ball would have been chosen before.
This is the condition `h`.

Finally, the last ball is chosen after all the other ones, meaning that `h` can be strengthened
by keeping only one side of the alternative in `hlast`.
-/
structure Besicovitch.SatelliteConfig (α : Type*) [MetricSpace α] (N : ℕ) (τ : ℝ) where
  c : Fin N.succ → α
  r : Fin N.succ → ℝ
  rpos : ∀ i, 0 < r i
  h : Pairwise fun i j =>
    r i ≤ dist (c i) (c j) ∧ r j ≤ τ * r i ∨ r j ≤ dist (c j) (c i) ∧ r i ≤ τ * r j
  hlast : ∀ i < last N, r i ≤ dist (c i) (c (last N)) ∧ r (last N) ≤ τ * r i
  inter : ∀ i < last N, dist (c i) (c (last N)) ≤ r i + r (last N)


/-- Extension for the `positivity` tactic: `Besicovitch.SatelliteConfig.r`. -/
@[positivity Besicovitch.SatelliteConfig.r _ _]
def evalBesicovitchSatelliteConfigR : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@Besicovitch.SatelliteConfig.r $β $inst $N $τ $self $i) =>
    assertInstancesCommute
    return .positive q(Besicovitch.SatelliteConfig.rpos $self $i)
  | _, _, _ => throwError "not Besicovitch.SatelliteConfig.r"


/-- A metric space has the Besicovitch covering property if there exist `N` and `τ > 1` such that
there are no satellite configuration of parameter `τ` with `N+1` points. This is the condition that
guarantees that the measurable Besicovitch covering theorem holds. It is satisfied by
finite-dimensional real vector spaces. -/
class HasBesicovitchCovering (α : Type*) [MetricSpace α] : Prop where
  no_satelliteConfig : ∃ (N : ℕ) (τ : ℝ), 1 < τ ∧ IsEmpty (Besicovitch.SatelliteConfig α N τ)


/-- There is always a satellite configuration with a single point. -/
instance Besicovitch.SatelliteConfig.instInhabited {α : Type*} {τ : ℝ}
    [Inhabited α] [MetricSpace α] : Inhabited (Besicovitch.SatelliteConfig α 0 τ) :=
  ⟨{  c := default
      r := fun _ => 1
      rpos := fun _ => zero_lt_one
      h := fun i j hij => (hij (Subsingleton.elim (α := Fin 1) i j)).elim
      hlast := fun i hi => by
        /-
          α : Type u_1
          τ : Real
          inst✝¹ : Inhabited α
          inst✝ : MetricSpace α
          i : Fin (HAdd.hAdd 0 1)
          hi : LT.lt i (Fin.last 0)
          ⊢ And (LE.le ((fun x => 1) i) (Dist.dist (Inhabited.default i) (Inhabited.defa …
        -/
        rw [Subsingleton.elim (α := Fin 1) i (last 0)] at hi; exact (lt_irrefl _ hi).elim
                                                              /-
                                                                🎉 no goals
                                                              -/
      inter := fun i hi => by
        /-
          α : Type u_1
          τ : Real
          inst✝¹ : Inhabited α
          inst✝ : MetricSpace α
          i : Fin (HAdd.hAdd 0 1)
          hi : LT.lt i (Fin.last 0)
          ⊢ LE.le (Dist.dist (Inhabited.default i) (Inhabited.default (Fin.last 0))) (HA …
        -/
        rw [Subsingleton.elim (α := Fin 1) i (last 0)] at hi; exact (lt_irrefl _ hi).elim }⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem inter' (i : Fin N.succ) : dist (a.c i) (a.c (last N)) ≤ a.r i + a.r (last N) := by
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    N : Nat
    τ : Real
    a : Besicovitch.SatelliteConfig α N τ
    i : Fin N.succ
    ⊢ LE.le (Dist.dist (a.c i) (a.c (Fin.last N))) (HAdd.hAdd (a.r i) (a.r (Fin.la …
  -/
  rcases lt_or_le i (last N) with (H | H)
    /-
      case inl
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      H : LT.lt i (Fin.last N)
      ⊢ LE.le (Dist.dist (a.c i) (a.c (Fin.last N))) (HAdd.hAdd (a.r i) (a.r (Fin.la …
    -/
  · exact a.inter i H
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      H : LE.le (Fin.last N) i
      ⊢ LE.le (Dist.dist (a.c i) (a.c (Fin.last N))) (HAdd.hAdd (a.r i) (a.r (Fin.la …
    -/
  · have I : i = last N := top_le_iff.1 H
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      H : LE.le (Fin.last N) i
      I : Eq i (Fin.last N)
      ⊢ LE.le (Dist.dist (a.c i) (a.c (Fin.last N))) (HAdd.hAdd (a.r i) (a.r (Fin.la …
    -/
    have := (a.rpos (last N)).le
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      H : LE.le (Fin.last N) i
      I : Eq i (Fin.last N)
      this : LE.le 0 (a.r (Fin.last N))
      ⊢ LE.le (Dist.dist (a.c i) (a.c (Fin.last N))) (HAdd.hAdd (a.r i) (a.r (Fin.la …
    -/
    simp only [I, add_nonneg this this, dist_self]
    /-
      🎉 no goals
    -/


theorem hlast' (i : Fin N.succ) (h : 1 ≤ τ) : a.r (last N) ≤ τ * a.r i := by
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    N : Nat
    τ : Real
    a : Besicovitch.SatelliteConfig α N τ
    i : Fin N.succ
    h : LE.le 1 τ
    ⊢ LE.le (a.r (Fin.last N)) (HMul.hMul τ (a.r i))
  -/
  rcases lt_or_le i (last N) with (H | H)
    /-
      case inl
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      h : LE.le 1 τ
      H : LT.lt i (Fin.last N)
      ⊢ LE.le (a.r (Fin.last N)) (HMul.hMul τ (a.r i))
    -/
  · exact (a.hlast i H).2
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      h : LE.le 1 τ
      H : LE.le (Fin.last N) i
      ⊢ LE.le (a.r (Fin.last N)) (HMul.hMul τ (a.r i))
    -/
  · have : i = last N := top_le_iff.1 H
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      h : LE.le 1 τ
      H : LE.le (Fin.last N) i
      this : Eq i (Fin.last N)
      ⊢ LE.le (a.r (Fin.last N)) (HMul.hMul τ (a.r i))
    -/
    rw [this]
    /-
      case inr
      α : Type u_1
      inst✝ : MetricSpace α
      N : Nat
      τ : Real
      a : Besicovitch.SatelliteConfig α N τ
      i : Fin N.succ
      h : LE.le 1 τ
      H : LE.le (Fin.last N) i
      this : Eq i (Fin.last N)
      ⊢ LE.le (a.r (Fin.last N)) (HMul.hMul τ (a.r (Fin.last N)))
    -/
    exact le_mul_of_one_le_left (a.rpos _).le h
    /-
      🎉 no goals
    -/


/-- A ball package is a family of balls in a metric space with positive bounded radii. -/
structure BallPackage (β : Type*) (α : Type*) where
  c : β → α
  r : β → ℝ
  rpos : ∀ b, 0 < r b
  r_bound : ℝ
  r_le : ∀ b, r b ≤ r_bound


/-- The ball package made of unit balls. -/
def unitBallPackage (α : Type*) : BallPackage α α where
  c := id
  r _ := 1
  rpos _ := zero_lt_one
  r_bound := 1
  r_le _ := le_rfl


instance BallPackage.instInhabited (α : Type*) : Inhabited (BallPackage α α) :=
  ⟨unitBallPackage α⟩


/-- A Besicovitch tau-package is a family of balls in a metric space with positive bounded radii,
together with enough data to proceed with the Besicovitch greedy algorithm. We register this in
a single structure to make sure that all our constructions in this algorithm only depend on
one variable. -/
structure TauPackage (β : Type*) (α : Type*) extends BallPackage β α where
  τ : ℝ
  one_lt_tau : 1 < τ


instance TauPackage.instInhabited (α : Type*) : Inhabited (TauPackage α α) :=
  ⟨{ unitBallPackage α with
      τ := 2
      one_lt_tau := one_lt_two }⟩


/-- Choose inductively large balls with centers that are not contained in the union of already
chosen balls. This is a transfinite induction. -/
noncomputable def index : Ordinal.{u} → β
  | i =>
      -- `Z` is the set of points that are covered by already constructed balls
      let Z := ⋃ j : { j // j < i }, ball (p.c (index j)) (p.r (index j))
      -- `R` is the supremum of the radii of balls with centers not in `Z`
      let R := iSup fun b : { b : β // p.c b ∉ Z } => p.r b
      -- return an index `b` for which the center `c b` is not in `Z`, and the radius is at
      -- least `R / τ`, if such an index exists (and garbage otherwise).
      Classical.epsilon fun b : β => p.c b ∉ Z ∧ R ≤ p.τ * p.r b
  termination_by i => i
  /-
    a✝ : Ordinal.{u}
    i : Ordinal.{u} := a✝
    j : Subtype fun j => LT.lt j i
    ⊢ LT.lt (↑j) a✝
  -/
  decreasing_by exact j.2
  /-
    🎉 no goals
  -/


/-- The set of points that are covered by the union of balls selected at steps `< i`. -/
def iUnionUpTo (i : Ordinal.{u}) : Set α :=
  ⋃ j : { j // j < i }, ball (p.c (p.index j)) (p.r (p.index j))


theorem monotone_iUnionUpTo : Monotone p.iUnionUpTo := by
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    ⊢ Monotone p.iUnionUpTo
  -/
  intro i j hij
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    i j : Ordinal.{u}
    hij : LE.le i j
    ⊢ LE.le (p.iUnionUpTo i) (p.iUnionUpTo j)
  -/
  simp only [iUnionUpTo]
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    i j : Ordinal.{u}
    hij : LE.le i j
    ⊢ LE.le (Set.iUnion fun j => Metric.ball (p.c (p.index ↑j)) (p.r (p.index ↑j)) …
  -/
  exact iUnion_mono' fun r => ⟨⟨r, r.2.trans_le hij⟩, Subset.rfl⟩
  /-
    🎉 no goals
  -/


/-- Supremum of the radii of balls whose centers are not yet covered at step `i`. -/
def R (i : Ordinal.{u}) : ℝ :=
  iSup fun b : { b : β // p.c b ∉ p.iUnionUpTo i } => p.r b


/-- Group the balls into disjoint families, by assigning to a ball the smallest color for which
it does not intersect any already chosen ball of this color. -/
noncomputable def color : Ordinal.{u} → ℕ
  | i =>
    let A : Set ℕ :=
      ⋃ (j : { j // j < i })
        (_ : (closedBall (p.c (p.index j)) (p.r (p.index j)) ∩
          closedBall (p.c (p.index i)) (p.r (p.index i))).Nonempty), {color j}
    sInf (univ \ A)
  termination_by i => i
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    a✝ : Ordinal.{u}
    i : Ordinal.{u} := a✝
    j : Subtype fun j => LT.lt j i
    x✝ : (Inter.inter (Metric.closedBall (p.c (p.index ↑j)) (p.r (p.index ↑j))) (M …
    ⊢ LT.lt (↑j) a✝
  -/
  decreasing_by exact j.2
  /-
    🎉 no goals
  -/


/-- `p.lastStep` is the first ordinal where the construction stops making sense, i.e., `f` returns
garbage since there is no point left to be chosen. We will only use ordinals before this step. -/
def lastStep : Ordinal.{u} :=
  sInf {i | ¬∃ b : β, p.c b ∉ p.iUnionUpTo i ∧ p.R i ≤ p.τ * p.r b}


theorem lastStep_nonempty :
    {i | ¬∃ b : β, p.c b ∉ p.iUnionUpTo i ∧ p.R i ≤ p.τ * p.r b}.Nonempty := by
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    ⊢ (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnionUpTo  …
  -/
  by_contra h
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    h : Not (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnio …
    ⊢ False
  -/
  suffices H : Function.Injective p.index from not_injective_of_ordinal p.index H
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    h : Not (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnio …
    ⊢ Function.Injective p.index
  -/
  intro x y hxy
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    h : Not (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnio …
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    ⊢ Eq x y
  -/
  wlog x_le_y : x ≤ y generalizing x y
    /-
      case inr
      α : Type u_1
      inst✝¹ : MetricSpace α
      β : Type u
      inst✝ : Nonempty β
      p : Besicovitch.TauPackage β α
      h : Not (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnio …
      x y : Ordinal.{u}
      hxy : Eq (p.index x) (p.index y)
      this : ∀ ⦃x y : Ordinal.{u}⦄, Eq (p.index x) (p.index y) → LE.le x y → Eq x y
      x_le_y : Not (LE.le x y)
      ⊢ Eq x y
    -/
  · exact (this hxy.symm (le_of_not_le x_le_y)).symm
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    h : Not (setOf fun i => Not (Exists fun b => And (Not (Membership.mem (p.iUnio …
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    x_le_y : LE.le x y
    ⊢ Eq x y
  -/
  rcases eq_or_lt_of_le x_le_y with (rfl | H); · rfl
                                                 /-
                                                   🎉 no goals
                                                 -/
  simp only [nonempty_def, not_exists, exists_prop, not_and, not_lt, not_le, mem_setOf_eq,
    not_forall] at h
  /-
    case inr
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    x_le_y : LE.le x y
    H : LT.lt x y
    h : ∀ (x : Ordinal.{u}), Exists fun x_1 => And (Not (Membership.mem (p.iUnionU …
    ⊢ Eq x y
  -/
  specialize h y
  have A : p.c (p.index y) ∉ p.iUnionUpTo y := by
    have :
        p.index y =
          Classical.epsilon fun b : β => p.c b ∉ p.iUnionUpTo y ∧ p.R y ≤ p.τ * p.r b := by
      rw [TauPackage.index]; rfl
    rw [this]
    exact (Classical.epsilon_spec h).1
  simp only [iUnionUpTo, not_exists, exists_prop, mem_iUnion, mem_closedBall, not_and, not_le,
    Subtype.exists, Subtype.coe_mk] at A
  /-
    case inr
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    x_le_y : LE.le x y
    H : LT.lt x y
    h : Exists fun x => And (Not (Membership.mem (p.iUnionUpTo y) (p.c x))) (LE.le …
    A : ∀ (x : Ordinal.{u}), LT.lt x y → Not (Membership.mem (Metric.ball (p.c (p. …
    ⊢ Eq x y
  -/
  specialize A x H
  /-
    case inr
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    x_le_y : LE.le x y
    H : LT.lt x y
    h : Exists fun x => And (Not (Membership.mem (p.iUnionUpTo y) (p.c x))) (LE.le …
    A : Not (Membership.mem (Metric.ball (p.c (p.index x)) (p.r (p.index x))) (p.c …
    ⊢ Eq x y
  -/
  simp? [hxy] at A says simp only [hxy, mem_ball, dist_self, not_lt] at A
  /-
    case inr
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x y : Ordinal.{u}
    hxy : Eq (p.index x) (p.index y)
    x_le_y : LE.le x y
    H : LT.lt x y
    h : Exists fun x => And (Not (Membership.mem (p.iUnionUpTo y) (p.c x))) (LE.le …
    A : LE.le (p.r (p.index y)) 0
    ⊢ Eq x y
  -/
  exact (lt_irrefl _ ((p.rpos (p.index y)).trans_le A)).elim
  /-
    🎉 no goals
  -/


/-- Every point is covered by chosen balls, before `p.lastStep`. -/
theorem mem_iUnionUpTo_lastStep (x : β) : p.c x ∈ p.iUnionUpTo p.lastStep := by
  have A : ∀ z : β, p.c z ∈ p.iUnionUpTo p.lastStep ∨ p.τ * p.r z < p.R p.lastStep := by
    have : p.lastStep ∈ {i | ¬∃ b : β, p.c b ∉ p.iUnionUpTo i ∧ p.R i ≤ p.τ * p.r b} :=
      csInf_mem p.lastStep_nonempty
    simpa only [not_exists, mem_setOf_eq, not_and_or, not_le, not_not_mem]
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x : β
    A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
    ⊢ Membership.mem (p.iUnionUpTo p.lastStep) (p.c x)
  -/
  by_contra h
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x : β
    A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
    h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
    ⊢ False
  -/
  rcases A x with (H | H); · exact h H
                             /-
                               🎉 no goals
                             -/
  have Rpos : 0 < p.R p.lastStep := by
    apply lt_trans (mul_pos (_root_.zero_lt_one.trans p.one_lt_tau) (p.rpos _)) H
  have B : p.τ⁻¹ * p.R p.lastStep < p.R p.lastStep := by
    conv_rhs => rw [← one_mul (p.R p.lastStep)]
    exact mul_lt_mul (inv_lt_one_of_one_lt₀ p.one_lt_tau) le_rfl Rpos zero_le_one
  obtain ⟨y, hy1, hy2⟩ : ∃ y, p.c y ∉ p.iUnionUpTo p.lastStep ∧ p.τ⁻¹ * p.R p.lastStep < p.r y := by
    have := exists_lt_of_lt_csSup ?_ B
    · simpa only [exists_prop, mem_range, exists_exists_and_eq_and, Subtype.exists,
      Subtype.coe_mk]
    rw [← image_univ, image_nonempty]
    exact ⟨⟨_, h⟩, mem_univ _⟩
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    x : β
    A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
    h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
    H : LT.lt (HMul.hMul p.τ (p.r x)) (p.R p.lastStep)
    Rpos : LT.lt 0 (p.R p.lastStep)
    B : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.R p.lastStep)
    y : β
    hy1 : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c y))
    hy2 : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.r y)
    ⊢ False
  -/
  rcases A y with (Hy | Hy)
    /-
      case inr.intro.intro.inl
      α : Type u_1
      inst✝¹ : MetricSpace α
      β : Type u
      inst✝ : Nonempty β
      p : Besicovitch.TauPackage β α
      x : β
      A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
      h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
      H : LT.lt (HMul.hMul p.τ (p.r x)) (p.R p.lastStep)
      Rpos : LT.lt 0 (p.R p.lastStep)
      B : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.R p.lastStep)
      y : β
      hy1 : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c y))
      hy2 : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.r y)
      Hy : Membership.mem (p.iUnionUpTo p.lastStep) (p.c y)
      ⊢ False
    -/
  · exact hy1 Hy
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.inr
      α : Type u_1
      inst✝¹ : MetricSpace α
      β : Type u
      inst✝ : Nonempty β
      p : Besicovitch.TauPackage β α
      x : β
      A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
      h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
      H : LT.lt (HMul.hMul p.τ (p.r x)) (p.R p.lastStep)
      Rpos : LT.lt 0 (p.R p.lastStep)
      B : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.R p.lastStep)
      y : β
      hy1 : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c y))
      hy2 : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.r y)
      Hy : LT.lt (HMul.hMul p.τ (p.r y)) (p.R p.lastStep)
      ⊢ False
    -/
  · rw [← div_eq_inv_mul] at hy2
    /-
      case inr.intro.intro.inr
      α : Type u_1
      inst✝¹ : MetricSpace α
      β : Type u
      inst✝ : Nonempty β
      p : Besicovitch.TauPackage β α
      x : β
      A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
      h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
      H : LT.lt (HMul.hMul p.τ (p.r x)) (p.R p.lastStep)
      Rpos : LT.lt 0 (p.R p.lastStep)
      B : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.R p.lastStep)
      y : β
      hy1 : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c y))
      hy2 : LT.lt (HDiv.hDiv (p.R p.lastStep) p.τ) (p.r y)
      Hy : LT.lt (HMul.hMul p.τ (p.r y)) (p.R p.lastStep)
      ⊢ False
    -/
    have := (div_le_iff₀' (_root_.zero_lt_one.trans p.one_lt_tau)).1 hy2.le
    /-
      case inr.intro.intro.inr
      α : Type u_1
      inst✝¹ : MetricSpace α
      β : Type u
      inst✝ : Nonempty β
      p : Besicovitch.TauPackage β α
      x : β
      A : ∀ (z : β), Or (Membership.mem (p.iUnionUpTo p.lastStep) (p.c z)) (LT.lt (H …
      h : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c x))
      H : LT.lt (HMul.hMul p.τ (p.r x)) (p.R p.lastStep)
      Rpos : LT.lt 0 (p.R p.lastStep)
      B : LT.lt (HMul.hMul (Inv.inv p.τ) (p.R p.lastStep)) (p.R p.lastStep)
      y : β
      hy1 : Not (Membership.mem (p.iUnionUpTo p.lastStep) (p.c y))
      hy2 : LT.lt (HDiv.hDiv (p.R p.lastStep) p.τ) (p.r y)
      Hy : LT.lt (HMul.hMul p.τ (p.r y)) (p.R p.lastStep)
      this : LE.le (p.R p.lastStep) (HMul.hMul p.τ (p.r y))
      ⊢ False
    -/
    exact lt_irrefl _ (Hy.trans_le this)
    /-
      🎉 no goals
    -/


/-- If there are no configurations of satellites with `N+1` points, one never uses more than `N`
distinct families in the Besicovitch inductive construction. -/
theorem color_lt {i : Ordinal.{u}} (hi : i < p.lastStep) {N : ℕ}
    (hN : IsEmpty (SatelliteConfig α N p.τ)) : p.color i < N := by
  /- By contradiction, consider the first ordinal `i` for which one would have `p.color i = N`.
    Choose for each `k < N` a ball with color `k` that intersects the ball at color `i`
    (there is such a ball, otherwise one would have used the color `k` and not `N`).
    Then this family of `N+1` balls forms a satellite configuration, which is forbidden by
    the assumption `hN`. -/
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    i : Ordinal.{u}
    hi : LT.lt i p.lastStep
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    ⊢ LT.lt (p.color i) N
  -/
  induction' i using Ordinal.induction with i IH
  let A : Set ℕ :=
    ⋃ (j : { j // j < i })
      (_ : (closedBall (p.c (p.index j)) (p.r (p.index j)) ∩
        closedBall (p.c (p.index i)) (p.r (p.index i))).Nonempty),
      {p.color j}
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    ⊢ LT.lt (p.color i) N
  -/
  have color_i : p.color i = sInf (univ \ A) := by rw [color]
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    color_i : Eq (p.color i) (InfSet.sInf (SDiff.sdiff Set.univ A))
    ⊢ LT.lt (p.color i) N
  -/
  rw [color_i]
  have N_mem : N ∈ univ \ A := by
    simp only [A, not_exists, true_and, exists_prop, mem_iUnion, mem_singleton_iff,
      mem_closedBall, not_and, mem_univ, mem_diff, Subtype.exists, Subtype.coe_mk]
    intro j ji _
    exact (IH j ji (ji.trans hi)).ne'
  suffices sInf (univ \ A) ≠ N by
    rcases (csInf_le (OrderBot.bddBelow (univ \ A)) N_mem).lt_or_eq with (H | H)
    · exact H
    · exact (this H).elim
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    color_i : Eq (p.color i) (InfSet.sInf (SDiff.sdiff Set.univ A))
    N_mem : Membership.mem (SDiff.sdiff Set.univ A) N
    ⊢ Ne (InfSet.sInf (SDiff.sdiff Set.univ A)) N
  -/
  intro Inf_eq_N
  have :
    ∀ k, k < N → ∃ j, j < i ∧
      (closedBall (p.c (p.index j)) (p.r (p.index j)) ∩
        closedBall (p.c (p.index i)) (p.r (p.index i))).Nonempty ∧ k = p.color j := by
    intro k hk
    rw [← Inf_eq_N] at hk
    have : k ∈ A := by
      simpa only [true_and, mem_univ, Classical.not_not, mem_diff] using
        Nat.not_mem_of_lt_sInf hk
    simp only [mem_iUnion, mem_singleton_iff, exists_prop, Subtype.exists, exists_and_right,
      and_assoc] at this
    simpa only [A, exists_prop, mem_iUnion, mem_singleton_iff, mem_closedBall, Subtype.exists,
      Subtype.coe_mk]
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    color_i : Eq (p.color i) (InfSet.sInf (SDiff.sdiff Set.univ A))
    N_mem : Membership.mem (SDiff.sdiff Set.univ A) N
    Inf_eq_N : Eq (InfSet.sInf (SDiff.sdiff Set.univ A)) N
    this : ∀ (k : Nat), LT.lt k N → Exists fun j => And (LT.lt j i) (And (Inter.in …
    ⊢ False
  -/
  choose! g hg using this
  -- Choose for each `k < N` an ordinal `G k < i` giving a ball of color `k` intersecting
  -- the last ball.
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    color_i : Eq (p.color i) (InfSet.sInf (SDiff.sdiff Set.univ A))
    N_mem : Membership.mem (SDiff.sdiff Set.univ A) N
    Inf_eq_N : Eq (InfSet.sInf (SDiff.sdiff Set.univ A)) N
    g : Nat → Ordinal.{u}
    hg : ∀ (k : Nat), LT.lt k N → And (LT.lt (g k) i) (And (Inter.inter (Metric.cl …
    ⊢ False
  -/
  let G : ℕ → Ordinal := fun n => if n = N then i else g n
  have color_G : ∀ n, n ≤ N → p.color (G n) = n := by
    intro n hn
    rcases hn.eq_or_lt with (rfl | H)
    · simp only [G]; simp only [color_i, Inf_eq_N, if_true, eq_self_iff_true]
    · simp only [G]; simp only [H.ne, (hg n H).right.right.symm, if_false]
  have G_lt_last : ∀ n, n ≤ N → G n < p.lastStep := by
    intro n hn
    rcases hn.eq_or_lt with (rfl | H)
    · simp only [G]; simp only [hi, if_true, eq_self_iff_true]
    · simp only [G]; simp only [H.ne, (hg n H).left.trans hi, if_false]
  have fGn :
      ∀ n, n ≤ N →
        p.c (p.index (G n)) ∉ p.iUnionUpTo (G n) ∧ p.R (G n) ≤ p.τ * p.r (p.index (G n)) := by
    intro n hn
    have :
      p.index (G n) =
        Classical.epsilon fun t => p.c t ∉ p.iUnionUpTo (G n) ∧ p.R (G n) ≤ p.τ * p.r t := by
      rw [index]; rfl
    rw [this]
    have : ∃ t, p.c t ∉ p.iUnionUpTo (G n) ∧ p.R (G n) ≤ p.τ * p.r t := by
      simpa only [not_exists, exists_prop, not_and, not_lt, not_le, mem_setOf_eq, not_forall] using
        not_mem_of_lt_csInf (G_lt_last n hn) (OrderBot.bddBelow _)
    exact Classical.epsilon_spec this
  -- the balls with indices `G k` satisfy the characteristic property of satellite configurations.
  have Gab :
    ∀ a b : Fin (Nat.succ N),
      G a < G b →
        p.r (p.index (G a)) ≤ dist (p.c (p.index (G a))) (p.c (p.index (G b))) ∧
          p.r (p.index (G b)) ≤ p.τ * p.r (p.index (G a)) := by
    intro a b G_lt
    have ha : (a : ℕ) ≤ N := Nat.lt_succ_iff.1 a.2
    have hb : (b : ℕ) ≤ N := Nat.lt_succ_iff.1 b.2
    constructor
    · have := (fGn b hb).1
      simp only [iUnionUpTo, not_exists, exists_prop, mem_iUnion, mem_closedBall, not_and, not_le,
        Subtype.exists, Subtype.coe_mk] at this
      simpa only [dist_comm, mem_ball, not_lt] using this (G a) G_lt
    · apply le_trans _ (fGn a ha).2
      have B : p.c (p.index (G b)) ∉ p.iUnionUpTo (G a) := by
        intro H; exact (fGn b hb).1 (p.monotone_iUnionUpTo G_lt.le H)
      let b' : { t // p.c t ∉ p.iUnionUpTo (G a) } := ⟨p.index (G b), B⟩
      apply @le_ciSup _ _ _ (fun t : { t // p.c t ∉ p.iUnionUpTo (G a) } => p.r t) _ b'
      refine ⟨p.r_bound, fun t ht => ?_⟩
      simp only [exists_prop, mem_range, Subtype.exists, Subtype.coe_mk] at ht
      rcases ht with ⟨u, hu⟩
      rw [← hu.2]
      exact p.r_le _
  -- therefore, one may use them to construct a satellite configuration with `N+1` points
  let sc : SatelliteConfig α N p.τ :=
    { c := fun k => p.c (p.index (G k))
      r := fun k => p.r (p.index (G k))
      rpos := fun k => p.rpos (p.index (G k))
      h := by
        intro a b a_ne_b
        wlog G_le : G a ≤ G b generalizing a b
        · exact (this a_ne_b.symm (le_of_not_le G_le)).symm
        have G_lt : G a < G b := by
          rcases G_le.lt_or_eq with (H | H); · exact H
          have A : (a : ℕ) ≠ b := Fin.val_injective.ne a_ne_b
          rw [← color_G a (Nat.lt_succ_iff.1 a.2), ← color_G b (Nat.lt_succ_iff.1 b.2), H] at A
          exact (A rfl).elim
        exact Or.inl (Gab a b G_lt)
      hlast := by
        intro a ha
        have I : (a : ℕ) < N := ha
        have : G a < G (Fin.last N) := by dsimp; simp [G, I.ne, (hg a I).1]
        exact Gab _ _ this
      inter := by
        intro a ha
        have I : (a : ℕ) < N := ha
        have J : G (Fin.last N) = i := by dsimp; simp only [G, if_true, eq_self_iff_true]
        have K : G a = g a := by dsimp [G]; simp [I.ne, (hg a I).1]
        convert dist_le_add_of_nonempty_closedBall_inter_closedBall (hg _ I).2.1 }
  -- this is a contradiction
  /-
    case h
    α : Type u_1
    inst✝¹ : MetricSpace α
    β : Type u
    inst✝ : Nonempty β
    p : Besicovitch.TauPackage β α
    N : Nat
    hN : IsEmpty (Besicovitch.SatelliteConfig α N p.τ)
    i : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k i → LT.lt k p.lastStep → LT.lt (p.color k) N
    hi : LT.lt i p.lastStep
    A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
    color_i : Eq (p.color i) (InfSet.sInf (SDiff.sdiff Set.univ A))
    N_mem : Membership.mem (SDiff.sdiff Set.univ A) N
    Inf_eq_N : Eq (InfSet.sInf (SDiff.sdiff Set.univ A)) N
    g : Nat → Ordinal.{u}
    hg : ∀ (k : Nat), LT.lt k N → And (LT.lt (g k) i) (And (Inter.inter (Metric.cl …
    G : Nat → Ordinal.{u} := fun n => ite (Eq n N) i (g n)
    color_G : ∀ (n : Nat), LE.le n N → Eq (p.color (G n)) n
    G_lt_last : ∀ (n : Nat), LE.le n N → LT.lt (G n) p.lastStep
    fGn : ∀ (n : Nat), LE.le n N → And (Not (Membership.mem (p.iUnionUpTo (G n)) ( …
    Gab : ∀ (a b : Fin N.succ), LT.lt (G ↑a) (G ↑b) → And (LE.le (p.r (p.index (G  …
    sc : Besicovitch.SatelliteConfig α N p.τ := { c := fun k => p.c (p.index (G ↑k …
    ⊢ False
  -/
  exact hN.false sc
  /-
    🎉 no goals
  -/


/-- The topological Besicovitch covering theorem: there exist finitely many families of disjoint
balls covering all the centers in a package. More specifically, one can use `N` families if there
are no satellite configurations with `N+1` points. -/
theorem exist_disjoint_covering_families {N : ℕ} {τ : ℝ} (hτ : 1 < τ)
    (hN : IsEmpty (SatelliteConfig α N τ)) (q : BallPackage β α) :
    ∃ s : Fin N → Set β,
      (∀ i : Fin N, (s i).PairwiseDisjoint fun j => closedBall (q.c j) (q.r j)) ∧
        range q.c ⊆ ⋃ i : Fin N, ⋃ j ∈ s i, ball (q.c j) (q.r j) := by
  -- first exclude the trivial case where `β` is empty (we need non-emptiness for the transfinite
  -- induction, to be able to choose garbage when there is no point left).
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    β : Type u
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    q : Besicovitch.BallPackage β α
    ⊢ Exists fun s => And (∀ (i : Fin N), (s i).PairwiseDisjoint fun j => Metric.c …
  -/
  cases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : IsEmpty β
      ⊢ Exists fun s => And (∀ (i : Fin N), (s i).PairwiseDisjoint fun j => Metric.c …
    -/
  · refine ⟨fun _ => ∅, fun _ => pairwiseDisjoint_empty, ?_⟩
    /-
      case inl
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : IsEmpty β
      ⊢ HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => Se …
    -/
    rw [← image_univ, eq_empty_of_isEmpty (univ : Set β)]
    /-
      case inl
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : IsEmpty β
      ⊢ HasSubset.Subset (Set.image q.c EmptyCollection.emptyCollection) (Set.iUnion …
    -/
    simp
    /-
      🎉 no goals
    -/
  -- Now, assume `β` is nonempty.
  let p : TauPackage β α :=
    { q with
      τ
      one_lt_tau := hτ }
  -- we use for `s i` the balls of color `i`.
  let s := fun i : Fin N =>
    ⋃ (k : Ordinal.{u}) (_ : k < p.lastStep) (_ : p.color k = i), ({p.index k} : Set β)
  /-
    case inr
    α : Type u_1
    inst✝ : MetricSpace α
    β : Type u
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    q : Besicovitch.BallPackage β α
    h✝ : Nonempty β
    p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
    s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
    ⊢ Exists fun s => And (∀ (i : Fin N), (s i).PairwiseDisjoint fun j => Metric.c …
  -/
  refine ⟨s, fun i => ?_, ?_⟩
  · -- show that balls of the same color are disjoint
    /-
      case inr.refine_1
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      ⊢ (s i).PairwiseDisjoint fun j => Metric.closedBall (q.c j) (q.r j)
    -/
    intro x hx y hy x_ne_y
    obtain ⟨jx, jx_lt, jxi, rfl⟩ :
      ∃ jx : Ordinal, jx < p.lastStep ∧ p.color jx = i ∧ x = p.index jx := by
      simpa only [s, exists_prop, mem_iUnion, mem_singleton_iff] using hx
    obtain ⟨jy, jy_lt, jyi, rfl⟩ :
      ∃ jy : Ordinal, jy < p.lastStep ∧ p.color jy = i ∧ y = p.index jy := by
      simpa only [s, exists_prop, mem_iUnion, mem_singleton_iff] using hy
    /-
      case inr.refine_1.intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      jx : Ordinal.{u}
      jx_lt : LT.lt jx p.lastStep
      jxi : Eq (p.color jx) ↑i
      hx : Membership.mem (s i) (p.index jx)
      jy : Ordinal.{u}
      jy_lt : LT.lt jy p.lastStep
      jyi : Eq (p.color jy) ↑i
      hy : Membership.mem (s i) (p.index jy)
      x_ne_y : Ne (p.index jx) (p.index jy)
      ⊢ Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j)) (p.inde …
    -/
    wlog jxy : jx ≤ jy generalizing jx jy
      /-
        case inr.refine_1.intro.intro.intro.intro.intro.intro.inr
        α : Type u_1
        inst✝ : MetricSpace α
        β : Type u
        N : Nat
        τ : Real
        hτ : LT.lt 1 τ
        hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
        q : Besicovitch.BallPackage β α
        h✝ : Nonempty β
        p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
        s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
        i : Fin N
        jx : Ordinal.{u}
        jx_lt : LT.lt jx p.lastStep
        jxi : Eq (p.color jx) ↑i
        hx : Membership.mem (s i) (p.index jx)
        jy : Ordinal.{u}
        jy_lt : LT.lt jy p.lastStep
        jyi : Eq (p.color jy) ↑i
        hy : Membership.mem (s i) (p.index jy)
        x_ne_y : Ne (p.index jx) (p.index jy)
        this : ∀ (jx : Ordinal.{u}), LT.lt jx p.lastStep → Eq (p.color jx) ↑i → Member …
        jxy : Not (LE.le jx jy)
        ⊢ Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j)) (p.inde …
      -/
    · exact (this jy jy_lt jyi hy jx jx_lt jxi hx x_ne_y.symm (le_of_not_le jxy)).symm
      /-
        🎉 no goals
      -/
    replace jxy : jx < jy := by
      rcases lt_or_eq_of_le jxy with (H | rfl); · { exact H }; · { exact (x_ne_y rfl).elim }
    let A : Set ℕ :=
      ⋃ (j : { j // j < jy })
        (_ : (closedBall (p.c (p.index j)) (p.r (p.index j)) ∩
          closedBall (p.c (p.index jy)) (p.r (p.index jy))).Nonempty),
        {p.color j}
    /-
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      jx : Ordinal.{u}
      jx_lt : LT.lt jx p.lastStep
      jxi : Eq (p.color jx) ↑i
      hx : Membership.mem (s i) (p.index jx)
      jy : Ordinal.{u}
      jy_lt : LT.lt jy p.lastStep
      jyi : Eq (p.color jy) ↑i
      hy : Membership.mem (s i) (p.index jy)
      x_ne_y : Ne (p.index jx) (p.index jy)
      jxy : LT.lt jx jy
      A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
      ⊢ Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j)) (p.inde …
    -/
    have color_j : p.color jy = sInf (univ \ A) := by rw [TauPackage.color]
    have h : p.color jy ∈ univ \ A := by
      rw [color_j]
      apply csInf_mem
      refine ⟨N, ?_⟩
      simp only [A, not_exists, true_and, exists_prop, mem_iUnion, mem_singleton_iff, not_and,
        mem_univ, mem_diff, Subtype.exists, Subtype.coe_mk]
      intro k hk _
      exact (p.color_lt (hk.trans jy_lt) hN).ne'
    simp only [A, not_exists, true_and, exists_prop, mem_iUnion, mem_singleton_iff, not_and,
      mem_univ, mem_diff, Subtype.exists, Subtype.coe_mk] at h
    /-
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      jx : Ordinal.{u}
      jx_lt : LT.lt jx p.lastStep
      jxi : Eq (p.color jx) ↑i
      hx : Membership.mem (s i) (p.index jx)
      jy : Ordinal.{u}
      jy_lt : LT.lt jy p.lastStep
      jyi : Eq (p.color jy) ↑i
      hy : Membership.mem (s i) (p.index jy)
      x_ne_y : Ne (p.index jx) (p.index jy)
      jxy : LT.lt jx jy
      A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
      color_j : Eq (p.color jy) (InfSet.sInf (SDiff.sdiff Set.univ A))
      h : ∀ (x : Ordinal.{u}), LT.lt x jy → (Inter.inter (Metric.closedBall (q.c (p. …
      ⊢ Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j)) (p.inde …
    -/
    specialize h jx jxy
    /-
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      jx : Ordinal.{u}
      jx_lt : LT.lt jx p.lastStep
      jxi : Eq (p.color jx) ↑i
      hx : Membership.mem (s i) (p.index jx)
      jy : Ordinal.{u}
      jy_lt : LT.lt jy p.lastStep
      jyi : Eq (p.color jy) ↑i
      hy : Membership.mem (s i) (p.index jy)
      x_ne_y : Ne (p.index jx) (p.index jy)
      jxy : LT.lt jx jy
      A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
      color_j : Eq (p.color jy) (InfSet.sInf (SDiff.sdiff Set.univ A))
      h : (Inter.inter (Metric.closedBall (q.c (p.index jx)) (q.r (p.index jx))) (Me …
      ⊢ Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j)) (p.inde …
    -/
    contrapose! h
    /-
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      i : Fin N
      jx : Ordinal.{u}
      jx_lt : LT.lt jx p.lastStep
      jxi : Eq (p.color jx) ↑i
      hx : Membership.mem (s i) (p.index jx)
      jy : Ordinal.{u}
      jy_lt : LT.lt jy p.lastStep
      jyi : Eq (p.color jy) ↑i
      hy : Membership.mem (s i) (p.index jy)
      x_ne_y : Ne (p.index jx) (p.index jy)
      jxy : LT.lt jx jy
      A : Set Nat := Set.iUnion fun j => Set.iUnion fun x => Singleton.singleton (p. …
      color_j : Eq (p.color jy) (InfSet.sInf (SDiff.sdiff Set.univ A))
      h : Not (Function.onFun Disjoint (fun j => Metric.closedBall (q.c j) (q.r j))  …
      ⊢ And (Inter.inter (Metric.closedBall (q.c (p.index jx)) (q.r (p.index jx))) ( …
    -/
    simpa only [jxi, jyi, and_true, eq_self_iff_true, ← not_disjoint_iff_nonempty_inter] using h
    /-
      🎉 no goals
    -/
  · -- show that the balls of color at most `N` cover every center.
    /-
      case inr.refine_2
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      ⊢ HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => Se …
    -/
    refine range_subset_iff.2 fun b => ?_
    obtain ⟨a, ha⟩ :
      ∃ a : Ordinal, a < p.lastStep ∧ dist (p.c b) (p.c (p.index a)) < p.r (p.index a) := by
      simpa only [iUnionUpTo, exists_prop, mem_iUnion, mem_ball, Subtype.exists,
        Subtype.coe_mk] using p.mem_iUnionUpTo_lastStep b
    simp only [s, exists_prop, mem_iUnion, mem_ball, mem_singleton_iff, biUnion_and',
      exists_eq_left, iUnion_exists, exists_and_left]
    /-
      case inr.refine_2.intro
      α : Type u_1
      inst✝ : MetricSpace α
      β : Type u
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      q : Besicovitch.BallPackage β α
      h✝ : Nonempty β
      p : Besicovitch.TauPackage β α := { toBallPackage := q, τ := τ, one_lt_tau :=  …
      s : Fin N → Set β := fun i => Set.iUnion fun k => Set.iUnion fun x => Set.iUni …
      b : β
      a : Ordinal.{u}
      ha : And (LT.lt a p.lastStep) (LT.lt (Dist.dist (p.c b) (p.c (p.index a))) (p. …
      ⊢ Exists fun i => Exists fun i_1 => And (Eq (p.color i_1) ↑i) (And (LT.lt i_1  …
    -/
    exact ⟨⟨p.color a, p.color_lt ha.1 hN⟩, a, rfl, ha⟩
    /-
      🎉 no goals
    -/


/-- Consider, for each `x` in a set `s`, a radius `r x ∈ (0, 1]`. Then one can find finitely
many disjoint balls of the form `closedBall x (r x)` covering a proportion `1/(N+1)` of `s`, if
there are no satellite configurations with `N+1` points.
-/
theorem exist_finset_disjoint_balls_large_measure (μ : Measure α) [IsFiniteMeasure μ] {N : ℕ}
    {τ : ℝ} (hτ : 1 < τ) (hN : IsEmpty (SatelliteConfig α N τ)) (s : Set α) (r : α → ℝ)
    (rpos : ∀ x ∈ s, 0 < r x) (rle : ∀ x ∈ s, r x ≤ 1) :
    ∃ t : Finset α, ↑t ⊆ s ∧ μ (s \ ⋃ x ∈ t, closedBall x (r x)) ≤ N / (N + 1) * μ s ∧
      (t : Set α).PairwiseDisjoint fun x => closedBall x (r x) := by
  -- exclude the trivial case where `μ s = 0`.
  /-
    α : Type u_1
    inst✝⁴ : MetricSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    s : Set α
    r : α → Real
    rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
    rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
  -/
  rcases le_or_lt (μ s) 0 with (hμs | hμs)
    /-
      case inl
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LE.le (μ s) 0
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
    -/
  · have : μ s = 0 := le_bot_iff.1 hμs
    /-
      case inl
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LE.le (μ s) 0
      this : Eq (μ s) 0
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
    -/
    refine ⟨∅, by simp only [Finset.coe_empty, empty_subset], ?_, ?_⟩
    · simp only [this, Finset.not_mem_empty, diff_empty, iUnion_false, iUnion_empty,
        nonpos_iff_eq_zero, mul_zero]
      /-
        case inl.refine_2
        α : Type u_1
        inst✝⁴ : MetricSpace α
        inst✝³ : SecondCountableTopology α
        inst✝² : MeasurableSpace α
        inst✝¹ : OpensMeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        N : Nat
        τ : Real
        hτ : LT.lt 1 τ
        hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
        s : Set α
        r : α → Real
        rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
        rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
        hμs : LE.le (μ s) 0
        this : Eq (μ s) 0
        ⊢ (↑EmptyCollection.emptyCollection).PairwiseDisjoint fun x => Metric.closedBa …
      -/
    · simp only [Finset.coe_empty, pairwiseDisjoint_empty]
      /-
        🎉 no goals
      -/
  /-
    case inr
    α : Type u_1
    inst✝⁴ : MetricSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    s : Set α
    r : α → Real
    rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
    rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
    hμs : LT.lt 0 (μ s)
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
  -/
  cases isEmpty_or_nonempty α
    /-
      case inr.inl
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : IsEmpty α
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
    -/
  · simp only [eq_empty_of_isEmpty s, measure_empty] at hμs
    /-
      case inr.inl
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      h✝ : IsEmpty α
      hμs : LT.lt 0 0
      ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
    -/
    exact (lt_irrefl _ hμs).elim
    /-
      🎉 no goals
    -/
  have Npos : N ≠ 0 := by
    rintro rfl
    inhabit α
    exact not_isEmpty_of_nonempty _ hN
  -- introduce a measurable superset `o` with the same measure, for measure computations
  obtain ⟨o, so, omeas, μo⟩ : ∃ o : Set α, s ⊆ o ∧ MeasurableSet o ∧ μ o = μ s :=
    exists_measurable_superset μ s
  /- We will apply the topological Besicovitch theorem, giving `N` disjoint subfamilies of balls
    covering `s`. Among these, one of them covers a proportion at least `1/N` of `s`. A large
    enough finite subfamily will then cover a proportion at least `1/(N+1)`. -/
  let a : BallPackage s α :=
    { c := fun x => x
      r := fun x => r x
      rpos := fun x => rpos x x.2
      r_bound := 1
      r_le := fun x => rle x x.2 }
  /-
    case inr.inr.intro.intro.intro
    α : Type u_1
    inst✝⁴ : MetricSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    s : Set α
    r : α → Real
    rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
    rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
    hμs : LT.lt 0 (μ s)
    h✝ : Nonempty α
    Npos : Ne N 0
    o : Set α
    so : HasSubset.Subset s o
    omeas : MeasurableSet o
    μo : Eq (μ o) (μ s)
    a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
  -/
  rcases exist_disjoint_covering_families hτ hN a with ⟨u, hu, hu'⟩
  have u_count : ∀ i, (u i).Countable := by
    intro i
    refine (hu i).countable_of_nonempty_interior fun j _ => ?_
    have : (ball (j : α) (r j)).Nonempty := nonempty_ball.2 (a.rpos _)
    exact this.mono ball_subset_interior_closedBall
  /-
    case inr.inr.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : MetricSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    s : Set α
    r : α → Real
    rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
    rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
    hμs : LT.lt 0 (μ s)
    h✝ : Nonempty α
    Npos : Ne N 0
    o : Set α
    so : HasSubset.Subset s o
    omeas : MeasurableSet o
    μo : Eq (μ o) (μ s)
    a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
    u : Fin N → Set ↑s
    hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
    hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
    u_count : ∀ (i : Fin N), (u i).Countable
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
  -/
  let v : Fin N → Set α := fun i => ⋃ (x : s) (_ : x ∈ u i), closedBall x (r x)
  have A : s = ⋃ i : Fin N, s ∩ v i := by
    refine Subset.antisymm ?_ (iUnion_subset fun i => inter_subset_left)
    intro x hx
    obtain ⟨i, y, hxy, h'⟩ :
        ∃ (i : Fin N) (i_1 : ↥s), i_1 ∈ u i ∧ x ∈ ball (↑i_1) (r ↑i_1) := by
      have : x ∈ range a.c := by simpa only [a, Subtype.range_coe_subtype, setOf_mem_eq]
      simpa only [mem_iUnion, bex_def] using hu' this
    refine mem_iUnion.2 ⟨i, ⟨hx, ?_⟩⟩
    simp only [v, exists_prop, mem_iUnion, SetCoe.exists, exists_and_right, Subtype.coe_mk]
    exact ⟨y, ⟨y.2, by simpa only [Subtype.coe_eta]⟩, ball_subset_closedBall h'⟩
  have S : ∑ _i : Fin N, μ s / N ≤ ∑ i, μ (s ∩ v i) :=
    calc
      ∑ _i : Fin N, μ s / N = μ s := by
        simp only [Finset.card_fin, Finset.sum_const, nsmul_eq_mul]
        rw [ENNReal.mul_div_cancel]
        · simp only [Npos, Ne, Nat.cast_eq_zero, not_false_iff]
        · exact ENNReal.natCast_ne_top _
      _ ≤ ∑ i, μ (s ∩ v i) := by
        conv_lhs => rw [A]
        apply measure_iUnion_fintype_le
  -- choose an index `i` of a subfamily covering at least a proportion `1/N` of `s`.
  obtain ⟨i, -, hi⟩ : ∃ (i : Fin N), i ∈ Finset.univ ∧ μ s / N ≤ μ (s ∩ v i) := by
    apply ENNReal.exists_le_of_sum_le _ S
    exact ⟨⟨0, bot_lt_iff_ne_bot.2 Npos⟩, Finset.mem_univ _⟩
  replace hi : μ s / (N + 1) < μ (s ∩ v i) := by
    apply lt_of_lt_of_le _ hi
    apply (ENNReal.mul_lt_mul_left hμs.ne' (measure_lt_top μ s).ne).2
    rw [ENNReal.inv_lt_inv]
    conv_lhs => rw [← add_zero (N : ℝ≥0∞)]
    exact ENNReal.add_lt_add_left (ENNReal.natCast_ne_top N) zero_lt_one
  have B : μ (o ∩ v i) = ∑' x : u i, μ (o ∩ closedBall x (r x)) := by
    have : o ∩ v i = ⋃ (x : s) (_ : x ∈ u i), o ∩ closedBall x (r x) := by
      simp only [v, inter_iUnion]
    rw [this, measure_biUnion (u_count i)]
    · exact (hu i).mono fun k => inter_subset_right
    · exact fun b _ => omeas.inter measurableSet_closedBall
  -- A large enough finite subfamily of `u i` will also cover a proportion `> 1/(N+1)` of `s`.
  -- Since `s` might not be measurable, we express this in terms of the measurable superset `o`.
  obtain ⟨w, hw⟩ :
    ∃ w : Finset (u i), μ s / (N + 1) <
      ∑ x ∈ w, μ (o ∩ closedBall (x : α) (r (x : α))) := by
    have C : HasSum (fun x : u i => μ (o ∩ closedBall x (r x))) (μ (o ∩ v i)) := by
      rw [B]; exact ENNReal.summable.hasSum
    have : μ s / (N + 1) < μ (o ∩ v i) := hi.trans_le (measure_mono (inter_subset_inter_left _ so))
    exact ((tendsto_order.1 C).1 _ this).exists
  -- Bring back the finset `w i` of `↑(u i)` to a finset of `α`, and check that it works by design.
  /-
    case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁴ : MetricSpace α
    inst✝³ : SecondCountableTopology α
    inst✝² : MeasurableSpace α
    inst✝¹ : OpensMeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    s : Set α
    r : α → Real
    rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
    rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
    hμs : LT.lt 0 (μ s)
    h✝ : Nonempty α
    Npos : Ne N 0
    o : Set α
    so : HasSubset.Subset s o
    omeas : MeasurableSet o
    μo : Eq (μ o) (μ s)
    a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
    u : Fin N → Set ↑s
    hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
    hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
    u_count : ∀ (i : Fin N), (u i).Countable
    v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
    A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
    S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
    i : Fin N
    hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
    B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
    w : Finset ↑(u i)
    hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
    ⊢ Exists fun t => And (HasSubset.Subset (↑t) s) (And (LE.le (μ (SDiff.sdiff s  …
  -/
  refine ⟨Finset.image (fun x : u i => x) w, ?_, ?_, ?_⟩
  -- show that the finset is included in `s`.
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      ⊢ HasSubset.Subset (↑(Finset.image (fun x => ↑↑x) w)) s
    -/
  · simp only [image_subset_iff, Finset.coe_image]
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      ⊢ HasSubset.Subset (↑w) (Set.preimage (fun x => ↑↑x) s)
    -/
    intro y _
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      y : ↑(u i)
      a✝ : Membership.mem (↑w) y
      ⊢ Membership.mem (Set.preimage (fun x => ↑↑x) s) y
    -/
    simp only [Subtype.coe_prop, mem_preimage]
    /-
      🎉 no goals
    -/
  -- show that it covers a large enough proportion of `s`. For measure computations, we do not
  -- use `s` (which might not be measurable), but its measurable superset `o`. Since their measures
  -- are the same, this does not spoil the estimates
  · suffices H : μ (o \ ⋃ x ∈ w, closedBall (↑x) (r ↑x)) ≤ N / (N + 1) * μ s by
      rw [Finset.set_biUnion_finset_image]
      exact le_trans (measure_mono (diff_subset_diff so (Subset.refl _))) H
    rw [← diff_inter_self_eq_diff,
      measure_diff_le_iff_le_add _ inter_subset_right (measure_lt_top μ _).ne]
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      ⊢ LE.le (μ o) (HAdd.hAdd (μ (Inter.inter (Set.iUnion fun x => Set.iUnion fun h …
    -/
    swap
    · exact .inter
        (w.nullMeasurableSet_biUnion fun _ _ ↦ measurableSet_closedBall.nullMeasurableSet)
        omeas.nullMeasurableSet
    calc
      μ o = 1 / (N + 1) * μ s + N / (N + 1) * μ s := by
        rw [μo, ← add_mul, ENNReal.div_add_div_same, add_comm, ENNReal.div_self, one_mul] <;> simp
      _ ≤ μ ((⋃ x ∈ w, closedBall (↑x) (r ↑x)) ∩ o) + N / (N + 1) * μ s := by
        gcongr
        rw [one_div, mul_comm, ← div_eq_mul_inv]
        apply hw.le.trans (le_of_eq _)
        rw [← Finset.set_biUnion_coe, inter_comm _ o, inter_iUnion₂, Finset.set_biUnion_coe,
          measure_biUnion_finset]
        · have : (w : Set (u i)).PairwiseDisjoint
              fun b : u i => closedBall (b : α) (r (b : α)) := by
            intro k _ l _ hkl; exact hu i k.2 l.2 (Subtype.val_injective.ne hkl)
          exact this.mono fun k => inter_subset_right
        · intro b _
          apply omeas.inter measurableSet_closedBall
  -- show that the balls are disjoint
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      ⊢ (↑(Finset.image (fun x => ↑↑x) w)).PairwiseDisjoint fun x => Metric.closedBa …
    -/
  · intro k hk l hl hkl
    obtain ⟨k', _, rfl⟩ : ∃ k' : u i, k' ∈ w ∧ ↑k' = k := by
      simpa only [mem_image, Finset.mem_coe, Finset.coe_image] using hk
    obtain ⟨l', _, rfl⟩ : ∃ l' : u i, l' ∈ w ∧ ↑l' = l := by
      simpa only [mem_image, Finset.mem_coe, Finset.coe_image] using hl
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_3.intro.in …
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      k' : ↑(u i)
      left✝¹ : Membership.mem w k'
      hk : Membership.mem ↑(Finset.image (fun x => ↑↑x) w) ↑↑k'
      l' : ↑(u i)
      left✝ : Membership.mem w l'
      hl : Membership.mem ↑(Finset.image (fun x => ↑↑x) w) ↑↑l'
      hkl : Ne ↑↑k' ↑↑l'
      ⊢ Function.onFun Disjoint (fun x => Metric.closedBall x (r x)) ↑↑k' ↑↑l'
    -/
    have k'nel' : (k' : s) ≠ l' := by intro h; rw [h] at hkl; exact hkl rfl
    /-
      case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_3.intro.in …
      α : Type u_1
      inst✝⁴ : MetricSpace α
      inst✝³ : SecondCountableTopology α
      inst✝² : MeasurableSpace α
      inst✝¹ : OpensMeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      s : Set α
      r : α → Real
      rpos : ∀ (x : α), Membership.mem s x → LT.lt 0 (r x)
      rle : ∀ (x : α), Membership.mem s x → LE.le (r x) 1
      hμs : LT.lt 0 (μ s)
      h✝ : Nonempty α
      Npos : Ne N 0
      o : Set α
      so : HasSubset.Subset s o
      omeas : MeasurableSet o
      μo : Eq (μ o) (μ s)
      a : Besicovitch.BallPackage (↑s) α := { c := fun x => ↑x, r := fun x => r ↑x,  …
      u : Fin N → Set ↑s
      hu : ∀ (i : Fin N), (u i).PairwiseDisjoint fun j => Metric.closedBall (a.c j)  …
      hu' : HasSubset.Subset (Set.range a.c) (Set.iUnion fun i => Set.iUnion fun j = …
      u_count : ∀ (i : Fin N), (u i).Countable
      v : Fin N → Set α := fun i => Set.iUnion fun x => Set.iUnion fun x_1 => Metric …
      A : Eq s (Set.iUnion fun i => Inter.inter s (v i))
      S : LE.le (Finset.univ.sum fun _i => HDiv.hDiv (μ s) ↑N) (Finset.univ.sum fun  …
      i : Fin N
      hi : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (μ (Inter.inter s (v i)))
      B : Eq (μ (Inter.inter o (v i))) (tsum fun x => μ (Inter.inter o (Metric.close …
      w : Finset ↑(u i)
      hw : LT.lt (HDiv.hDiv (μ s) (HAdd.hAdd (↑N) 1)) (w.sum fun x => μ (Inter.inter …
      k' : ↑(u i)
      left✝¹ : Membership.mem w k'
      hk : Membership.mem ↑(Finset.image (fun x => ↑↑x) w) ↑↑k'
      l' : ↑(u i)
      left✝ : Membership.mem w l'
      hl : Membership.mem ↑(Finset.image (fun x => ↑↑x) w) ↑↑l'
      hkl : Ne ↑↑k' ↑↑l'
      k'nel' : Ne ↑k' ↑l'
      ⊢ Function.onFun Disjoint (fun x => Metric.closedBall x (r x)) ↑↑k' ↑↑l'
    -/
    exact hu i k'.2 l'.2 k'nel'
    /-
      🎉 no goals
    -/


/-- The **measurable Besicovitch covering theorem**. Assume that, for any `x` in a set `s`,
one is given a set of admissible closed balls centered at `x`, with arbitrarily small radii.
Then there exists a disjoint covering of almost all `s` by admissible closed balls centered at some
points of `s`.
This version requires that the underlying measure is finite, and that the space has the Besicovitch
covering property (which is satisfied for instance by normed real vector spaces). It expresses the
conclusion in a slightly awkward form (with a subset of `α × ℝ`) coming from the proof technique.
For a version assuming that the measure is sigma-finite,
see `exists_disjoint_closedBall_covering_ae_aux`.
For a version giving the conclusion in a nicer form, see `exists_disjoint_closedBall_covering_ae`.
-/
theorem exists_disjoint_closedBall_covering_ae_of_finiteMeasure_aux (μ : Measure α)
    [IsFiniteMeasure μ] (f : α → Set ℝ) (s : Set α)
    (hf : ∀ x ∈ s, ∀ δ > 0, (f x ∩ Ioo 0 δ).Nonempty) :
    ∃ t : Set (α × ℝ), t.Countable ∧ (∀ p ∈ t, p.1 ∈ s) ∧ (∀ p ∈ t, p.2 ∈ f p.1) ∧
      μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ t), closedBall p.1 p.2) = 0 ∧
        t.PairwiseDisjoint fun p => closedBall p.1 p.2 := by
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  rcases HasBesicovitchCovering.no_satelliteConfig (α := α) with ⟨N, τ, hτ, hN⟩
  /- Introduce a property `P` on finsets saying that we have a nice disjoint covering of a
      subset of `s` by admissible balls. -/
  let P : Finset (α × ℝ) → Prop := fun t =>
    ((t : Set (α × ℝ)).PairwiseDisjoint fun p => closedBall p.1 p.2) ∧
      (∀ p : α × ℝ, p ∈ t → p.1 ∈ s) ∧ ∀ p : α × ℝ, p ∈ t → p.2 ∈ f p.1
  /- Given a finite good covering of a subset `s`, one can find a larger finite good covering,
    covering additionally a proportion at least `1/(N+1)` of leftover points. This follows from
    `exist_finset_disjoint_balls_large_measure` applied to balls not intersecting the initial
    covering. -/
  have :
      ∀ t : Finset (α × ℝ), P t → ∃ u : Finset (α × ℝ), t ⊆ u ∧ P u ∧
        μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ u), closedBall p.1 p.2) ≤
          N / (N + 1) * μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ t), closedBall p.1 p.2) := by
    intro t ht
    set B := ⋃ (p : α × ℝ) (_ : p ∈ t), closedBall p.1 p.2 with hB
    have B_closed : IsClosed B := isClosed_biUnion_finset fun i _ => isClosed_ball
    set s' := s \ B
    have : ∀ x ∈ s', ∃ r ∈ f x ∩ Ioo 0 1, Disjoint B (closedBall x r) := by
      intro x hx
      have xs : x ∈ s := ((mem_diff x).1 hx).1
      rcases eq_empty_or_nonempty B with (hB | hB)
      · rcases hf x xs 1 zero_lt_one with ⟨r, hr, h'r⟩
        exact ⟨r, ⟨hr, h'r⟩, by simp only [hB, empty_disjoint]⟩
      · let r := infDist x B
        have : 0 < min r 1 :=
          lt_min ((B_closed.not_mem_iff_infDist_pos hB).1 ((mem_diff x).1 hx).2) zero_lt_one
        rcases hf x xs _ this with ⟨r, hr, h'r⟩
        refine ⟨r, ⟨hr, ⟨h'r.1, h'r.2.trans_le (min_le_right _ _)⟩⟩, ?_⟩
        rw [disjoint_comm]
        exact disjoint_closedBall_of_lt_infDist (h'r.2.trans_le (min_le_left _ _))
    choose! r hr using this
    obtain ⟨v, vs', hμv, hv⟩ :
      ∃ v : Finset α,
        ↑v ⊆ s' ∧
          μ (s' \ ⋃ x ∈ v, closedBall x (r x)) ≤ N / (N + 1) * μ s' ∧
            (v : Set α).PairwiseDisjoint fun x : α => closedBall x (r x) :=
      haveI rI : ∀ x ∈ s', r x ∈ Ioo (0 : ℝ) 1 := fun x hx => (hr x hx).1.2
      exist_finset_disjoint_balls_large_measure μ hτ hN s' r (fun x hx => (rI x hx).1) fun x hx =>
        (rI x hx).2.le
    refine ⟨t ∪ Finset.image (fun x => (x, r x)) v, Finset.subset_union_left, ⟨?_, ?_, ?_⟩, ?_⟩
    · simp only [Finset.coe_union, pairwiseDisjoint_union, ht.1, true_and, Finset.coe_image]
      constructor
      · intro p hp q hq hpq
        rcases (mem_image _ _ _).1 hp with ⟨p', p'v, rfl⟩
        rcases (mem_image _ _ _).1 hq with ⟨q', q'v, rfl⟩
        refine hv p'v q'v fun hp'q' => ?_
        rw [hp'q'] at hpq
        exact hpq rfl
      · intro p hp q hq hpq
        rcases (mem_image _ _ _).1 hq with ⟨q', q'v, rfl⟩
        apply disjoint_of_subset_left _ (hr q' (vs' q'v)).2
        rw [hB, ← Finset.set_biUnion_coe]
        exact subset_biUnion_of_mem (u := fun x : α × ℝ => closedBall x.1 x.2) hp
    · intro p hp
      rcases Finset.mem_union.1 hp with (h'p | h'p)
      · exact ht.2.1 p h'p
      · rcases Finset.mem_image.1 h'p with ⟨p', p'v, rfl⟩
        exact ((mem_diff _).1 (vs' (Finset.mem_coe.2 p'v))).1
    · intro p hp
      rcases Finset.mem_union.1 hp with (h'p | h'p)
      · exact ht.2.2 p h'p
      · rcases Finset.mem_image.1 h'p with ⟨p', p'v, rfl⟩
        exact (hr p' (vs' p'v)).1.1
    · convert hμv using 2
      rw [Finset.set_biUnion_union, ← diff_diff, Finset.set_biUnion_finset_image]
  /- Define `F` associating to a finite good covering the above enlarged good covering, covering
    a proportion `1/(N+1)` of leftover points. Iterating `F`, one will get larger and larger good
    coverings, missing in the end only a measure-zero set. -/
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
    this : ∀ (t : Finset (Prod α Real)), P t → Exists fun u => And (HasSubset.Subs …
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  choose! F hF using this
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
    F : Finset (Prod α Real) → Finset (Prod α Real)
    hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  let u n := F^[n] ∅
  have u_succ : ∀ n : ℕ, u n.succ = F (u n) := fun n => by
    simp only [u, Function.comp_apply, Function.iterate_succ']
  have Pu : ∀ n, P (u n) := by
    intro n
    induction' n with n IH
    · simp only [P, u, Prod.forall, id, Function.iterate_zero]
      simp only [Finset.not_mem_empty, IsEmpty.forall_iff, Finset.coe_empty, forall₂_true_iff,
        and_self_iff, pairwiseDisjoint_empty]
    · rw [u_succ]
      exact (hF (u n) IH).2.1
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
    F : Finset (Prod α Real) → Finset (Prod α Real)
    hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
    u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
    u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
    Pu : ∀ (n : Nat), P (u n)
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  refine ⟨⋃ n, u n, countable_iUnion fun n => (u n).countable_toSet, ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      ⊢ ∀ (p : Prod α Real), Membership.mem (Set.iUnion fun n => ↑(u n)) p → Members …
    -/
  · intro p hp
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      p : Prod α Real
      hp : Membership.mem (Set.iUnion fun n => ↑(u n)) p
      ⊢ Membership.mem s p.1
    -/
    rcases mem_iUnion.1 hp with ⟨n, hn⟩
    /-
      case intro.intro.intro.refine_1.intro
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      p : Prod α Real
      hp : Membership.mem (Set.iUnion fun n => ↑(u n)) p
      n : Nat
      hn : Membership.mem (↑(u n)) p
      ⊢ Membership.mem s p.1
    -/
    exact (Pu n).2.1 p (Finset.mem_coe.1 hn)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      ⊢ ∀ (p : Prod α Real), Membership.mem (Set.iUnion fun n => ↑(u n)) p → Members …
    -/
  · intro p hp
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      p : Prod α Real
      hp : Membership.mem (Set.iUnion fun n => ↑(u n)) p
      ⊢ Membership.mem (f p.1) p.2
    -/
    rcases mem_iUnion.1 hp with ⟨n, hn⟩
    /-
      case intro.intro.intro.refine_2.intro
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      p : Prod α Real
      hp : Membership.mem (Set.iUnion fun n => ↑(u n)) p
      n : Nat
      hn : Membership.mem (↑(u n)) p
      ⊢ Membership.mem (f p.1) p.2
    -/
    exact (Pu n).2.2 p (Finset.mem_coe.1 hn)
    /-
      🎉 no goals
    -/
  · have A :
      ∀ n,
        μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ ⋃ n : ℕ, (u n : Set (α × ℝ))), closedBall p.fst p.snd) ≤
          μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ u n), closedBall p.fst p.snd) := by
      intro n
      gcongr μ (s \ ?_)
      exact biUnion_subset_biUnion_left (subset_iUnion (fun i => (u i : Set (α × ℝ))) n)
    have B :
        ∀ n, μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ u n), closedBall p.fst p.snd) ≤
          (N / (N + 1) : ℝ≥0∞) ^ n * μ s := by
      intro n
      induction' n with n IH
      · simp only [u, le_refl, diff_empty, one_mul, iUnion_false, iUnion_empty, pow_zero,
          Function.iterate_zero, id, Finset.not_mem_empty]
      calc
        μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ u n.succ), closedBall p.fst p.snd) ≤
            N / (N + 1) * μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ u n), closedBall p.fst p.snd) := by
          rw [u_succ]; exact (hF (u n) (Pu n)).2.2
        _ ≤ (N / (N + 1) : ℝ≥0∞) ^ n.succ * μ s := by
          rw [pow_succ', mul_assoc]; exact mul_le_mul_left' IH _
    have C : Tendsto (fun n : ℕ => ((N : ℝ≥0∞) / (N + 1)) ^ n * μ s) atTop (𝓝 (0 * μ s)) := by
      apply ENNReal.Tendsto.mul_const _ (Or.inr (measure_lt_top μ s).ne)
      apply ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one
      rw [ENNReal.div_lt_iff, one_mul]
      · conv_lhs => rw [← add_zero (N : ℝ≥0∞)]
        exact ENNReal.add_lt_add_left (ENNReal.natCast_ne_top N) zero_lt_one
      · simp only [true_or, add_eq_zero, Ne, not_false_iff, one_ne_zero, and_false]
      · simp only [ENNReal.natCast_ne_top, Ne, not_false_iff, or_true]
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      B : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      C : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (HDiv.hDiv (↑N) (HAdd.hAdd ( …
      ⊢ Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.closedB …
    -/
    rw [zero_mul] at C
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      B : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      C : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (HDiv.hDiv (↑N) (HAdd.hAdd ( …
      ⊢ Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.closedB …
    -/
    apply le_bot_iff.1
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      A : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      B : ∀ (n : Nat), LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x …
      C : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (HDiv.hDiv (↑N) (HAdd.hAdd ( …
      ⊢ LE.le (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.clos …
    -/
    exact le_of_tendsto_of_tendsto' tendsto_const_nhds C fun n => (A n).trans (B n)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_4
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      ⊢ (Set.iUnion fun n => ↑(u n)).PairwiseDisjoint fun p => Metric.closedBall p.1 …
    -/
  · refine (pairwiseDisjoint_iUnion ?_).2 fun n => (Pu n).1
    /-
      case intro.intro.intro.refine_4
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      ⊢ Directed (fun x1 x2 => HasSubset.Subset x1 x2) fun n => ↑(u n)
    -/
    apply (monotone_nat_of_le_succ fun n => ?_).directed_le
    /-
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      n : Nat
      ⊢ LE.le ↑(u n) ↑(u (HAdd.hAdd n 1))
    -/
    rw [← Nat.succ_eq_add_one, u_succ]
    /-
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      hN : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      P : Finset (Prod α Real) → Prop := fun t => And ((↑t).PairwiseDisjoint fun p = …
      F : Finset (Prod α Real) → Finset (Prod α Real)
      hF : ∀ (t : Finset (Prod α Real)), P t → And (HasSubset.Subset t (F t)) (And ( …
      u : Nat → Finset (Prod α Real) := fun n => Nat.iterate F n EmptyCollection.emp …
      u_succ : ∀ (n : Nat), Eq (u n.succ) (F (u n))
      Pu : ∀ (n : Nat), P (u n)
      n : Nat
      ⊢ LE.le ↑(u n) ↑(F (u n))
    -/
    exact (hF (u n) (Pu n)).1
    /-
      🎉 no goals
    -/


/-- The measurable Besicovitch covering theorem. Assume that, for any `x` in a set `s`,
one is given a set of admissible closed balls centered at `x`, with arbitrarily small radii.
Then there exists a disjoint covering of almost all `s` by admissible closed balls centered at some
points of `s`.
This version requires that the underlying measure is sigma-finite, and that the space has the
Besicovitch covering property (which is satisfied for instance by normed real vector spaces).
It expresses the conclusion in a slightly awkward form (with a subset of `α × ℝ`) coming from the
proof technique.
For a version giving the conclusion in a nicer form, see `exists_disjoint_closedBall_covering_ae`.
-/
theorem exists_disjoint_closedBall_covering_ae_aux (μ : Measure α) [SFinite μ] (f : α → Set ℝ)
    (s : Set α) (hf : ∀ x ∈ s, ∀ δ > 0, (f x ∩ Ioo 0 δ).Nonempty) :
    ∃ t : Set (α × ℝ), t.Countable ∧ (∀ p ∈ t, p.1 ∈ s) ∧ (∀ p ∈ t, p.2 ∈ f p.1) ∧
      μ (s \ ⋃ (p : α × ℝ) (_ : p ∈ t), closedBall p.1 p.2) = 0 ∧
        t.PairwiseDisjoint fun p => closedBall p.1 p.2 := by
  /- This is deduced from the finite measure case, by using a finite measure with respect to which
    the initial sigma-finite measure is absolutely continuous. -/
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  rcases exists_isFiniteMeasure_absolutelyContinuous μ with ⟨ν, hν, hμν, -⟩
  rcases exists_disjoint_closedBall_covering_ae_of_finiteMeasure_aux ν f s hf with
    ⟨t, t_count, ts, tr, tν, tdisj⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    ν : MeasureTheory.Measure α
    hν : MeasureTheory.IsFiniteMeasure ν
    hμν : μ.AbsolutelyContinuous ν
    t : Set (Prod α Real)
    t_count : t.Countable
    ts : ∀ (p : Prod α Real), Membership.mem t p → Membership.mem s p.1
    tr : ∀ (p : Prod α Real), Membership.mem t p → Membership.mem (f p.1) p.2
    tν : Eq (ν (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.clos …
    tdisj : t.PairwiseDisjoint fun p => Metric.closedBall p.1 p.2
    ⊢ Exists fun t => And t.Countable (And (∀ (p : Prod α Real), Membership.mem t  …
  -/
  exact ⟨t, t_count, ts, tr, hμν tν, tdisj⟩
  /-
    🎉 no goals
  -/


/-- The measurable Besicovitch covering theorem. Assume that, for any `x` in a set `s`,
one is given a set of admissible closed balls centered at `x`, with arbitrarily small radii.
Then there exists a disjoint covering of almost all `s` by admissible closed balls centered at some
points of `s`. We can even require that the radius at `x` is bounded by a given function `R x`.
(Take `R = 1` if you don't need this additional feature).
This version requires that the underlying measure is sigma-finite, and that the space has the
Besicovitch covering property (which is satisfied for instance by normed real vector spaces).
-/
theorem exists_disjoint_closedBall_covering_ae (μ : Measure α) [SFinite μ] (f : α → Set ℝ)
    (s : Set α) (hf : ∀ x ∈ s, ∀ δ > 0, (f x ∩ Ioo 0 δ).Nonempty) (R : α → ℝ)
    (hR : ∀ x ∈ s, 0 < R x) :
    ∃ (t : Set α) (r : α → ℝ), t.Countable ∧ t ⊆ s ∧
      (∀ x ∈ t, r x ∈ f x ∩ Ioo 0 (R x)) ∧ μ (s \ ⋃ x ∈ t, closedBall x (r x)) = 0 ∧
        t.PairwiseDisjoint fun x => closedBall x (r x) := by
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → LT.lt 0 (R x)
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  let g x := f x ∩ Ioo 0 (R x)
  have hg : ∀ x ∈ s, ∀ δ > 0, (g x ∩ Ioo 0 δ).Nonempty := fun x hx δ δpos ↦ by
    rcases hf x hx (min δ (R x)) (lt_min δpos (hR x hx)) with ⟨r, hr⟩
    exact ⟨r, ⟨⟨hr.1, hr.2.1, hr.2.2.trans_le (min_le_right _ _)⟩,
      ⟨hr.2.1, hr.2.2.trans_le (min_le_left _ _)⟩⟩⟩
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → LT.lt 0 (R x)
    g : α → Set Real := fun x => Inter.inter (f x) (Set.Ioo 0 (R x))
    hg : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g …
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  rcases exists_disjoint_closedBall_covering_ae_aux μ g s hg with ⟨v, v_count, vs, vg, μv, v_disj⟩
  obtain ⟨r, t, rfl⟩ : ∃ (r : α → ℝ) (t : Set α), v = graphOn r t := by
    have I : ∀ p ∈ v, 0 ≤ p.2 := fun p hp => (vg p hp).2.1.le
    rw [exists_eq_graphOn]
    refine fun x hx y hy heq ↦ v_disj.eq hx hy <| not_disjoint_iff.2 ⟨x.1, ?_⟩
    simp [*]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → LT.lt 0 (R x)
    g : α → Set Real := fun x => Inter.inter (f x) (Set.Ioo 0 (R x))
    hg : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g …
    r : α → Real
    t : Set α
    v_count : (Set.graphOn r t).Countable
    vs : ∀ (p : Prod α Real), Membership.mem (Set.graphOn r t) p → Membership.mem  …
    vg : ∀ (p : Prod α Real), Membership.mem (Set.graphOn r t) p → Membership.mem  …
    μv : Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.clos …
    v_disj : (Set.graphOn r t).PairwiseDisjoint fun p => Metric.closedBall p.1 p.2
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  have hinj : InjOn (fun x ↦ (x, r x)) t := LeftInvOn.injOn (f₁' := Prod.fst) fun _ _ ↦ rfl
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → LT.lt 0 (R x)
    g : α → Set Real := fun x => Inter.inter (f x) (Set.Ioo 0 (R x))
    hg : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g …
    r : α → Real
    t : Set α
    v_count : (Set.graphOn r t).Countable
    vs : ∀ (p : Prod α Real), Membership.mem (Set.graphOn r t) p → Membership.mem  …
    vg : ∀ (p : Prod α Real), Membership.mem (Set.graphOn r t) p → Membership.mem  …
    μv : Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun x => Metric.clos …
    v_disj : (Set.graphOn r t).PairwiseDisjoint fun p => Metric.closedBall p.1 p.2
    hinj : Set.InjOn (fun x => { fst := x, snd := r x }) t
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  simp only [graphOn, forall_mem_image, biUnion_image, hinj.pairwiseDisjoint_image] at *
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → LT.lt 0 (R x)
    g : α → Set Real := fun x => Inter.inter (f x) (Set.Ioo 0 (R x))
    hg : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g …
    r : α → Real
    t : Set α
    v_count : (Set.image (fun x => { fst := x, snd := r x }) t).Countable
    hinj : Set.InjOn (fun x => { fst := x, snd := r x }) t
    vs : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem s x
    vg : ∀ ⦃x : α⦄, Membership.mem t x → Membership.mem (g x) (r x)
    μv : Eq (μ (SDiff.sdiff s (Set.iUnion fun y => Set.iUnion fun x => Metric.clos …
    v_disj : t.PairwiseDisjoint (Function.comp (fun p => Metric.closedBall p.1 p.2 …
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  exact ⟨t, r, countable_of_injective_of_countable_image hinj v_count, vs, vg, μv, v_disj⟩
  /-
    🎉 no goals
  -/


/-- In a space with the Besicovitch property, any set `s` can be covered with balls whose measures
add up to at most `μ s + ε`, for any positive `ε`. This works even if one restricts the set of
allowed radii around a point `x` to a set `f x` which accumulates at `0`. -/
theorem exists_closedBall_covering_tsum_measure_le (μ : Measure α) [SFinite μ]
    [Measure.OuterRegular μ] {ε : ℝ≥0∞} (hε : ε ≠ 0) (f : α → Set ℝ) (s : Set α)
    (hf : ∀ x ∈ s, ∀ δ > 0, (f x ∩ Ioo 0 δ).Nonempty) :
    ∃ (t : Set α) (r : α → ℝ), t.Countable ∧ t ⊆ s ∧ (∀ x ∈ t, r x ∈ f x) ∧
      (s ⊆ ⋃ x ∈ t, closedBall x (r x)) ∧ (∑' x : t, μ (closedBall x (r x))) ≤ μ s + ε := by
  /- For the proof, first cover almost all `s` with disjoint balls thanks to the usual Besicovitch
    theorem. Taking the balls included in a well-chosen open neighborhood `u` of `s`, one may
    ensure that their measures add at most to `μ s + ε / 2`. Let `s'` be the remaining set, of
    measure `0`. Applying the other version of Besicovitch, one may cover it with at most `N`
    disjoint subfamilies. Making sure that they are all included in a neighborhood `v` of `s'` of
    measure at most `ε / (2 N)`, the sum of their measures is at most `ε / 2`,
    completing the proof. -/
  obtain ⟨u, su, u_open, μu⟩ : ∃ U, U ⊇ s ∧ IsOpen U ∧ μ U ≤ μ s + ε / 2 :=
    Set.exists_isOpen_le_add _ _
      (by
        simpa only [or_false, Ne, ENNReal.div_eq_zero_iff, ENNReal.two_ne_top] using hε)
  have : ∀ x ∈ s, ∃ R > 0, ball x R ⊆ u := fun x hx =>
    Metric.mem_nhds_iff.1 (u_open.mem_nhds (su hx))
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    this : ∀ (x : α), Membership.mem s x → Exists fun R => And (GT.gt R 0) (HasSub …
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  choose! R hR using this
  obtain ⟨t0, r0, t0_count, t0s, hr0, μt0, t0_disj⟩ :
    ∃ (t0 : Set α) (r0 : α → ℝ), t0.Countable ∧ t0 ⊆ s ∧
      (∀ x ∈ t0, r0 x ∈ f x ∩ Ioo 0 (R x)) ∧ μ (s \ ⋃ x ∈ t0, closedBall x (r0 x)) = 0 ∧
        t0.PairwiseDisjoint fun x => closedBall x (r0 x) :=
    exists_disjoint_closedBall_covering_ae μ f s hf R fun x hx => (hR x hx).1
  -- we have constructed an almost everywhere covering of `s` by disjoint balls. Let `s'` be the
  -- remaining set.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
    t0 : Set α
    r0 : α → Real
    t0_count : t0.Countable
    t0s : HasSubset.Subset t0 s
    hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
    μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
    t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  let s' := s \ ⋃ x ∈ t0, closedBall x (r0 x)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
    t0 : Set α
    r0 : α → Real
    t0_count : t0.Countable
    t0s : HasSubset.Subset t0 s
    hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
    μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
    t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
    s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  have s's : s' ⊆ s := diff_subset
  obtain ⟨N, τ, hτ, H⟩ : ∃ N τ, 1 < τ ∧ IsEmpty (Besicovitch.SatelliteConfig α N τ) :=
    HasBesicovitchCovering.no_satelliteConfig
  obtain ⟨v, s'v, v_open, μv⟩ : ∃ v, v ⊇ s' ∧ IsOpen v ∧ μ v ≤ μ s' + ε / 2 / N :=
    Set.exists_isOpen_le_add _ _
      (by simp only [ne_eq, ENNReal.div_eq_zero_iff, hε, ENNReal.two_ne_top, or_self,
          ENNReal.natCast_ne_top, not_false_eq_true])
  have : ∀ x ∈ s', ∃ r1 ∈ f x ∩ Ioo (0 : ℝ) 1, closedBall x r1 ⊆ v := by
    intro x hx
    rcases Metric.mem_nhds_iff.1 (v_open.mem_nhds (s'v hx)) with ⟨r, rpos, hr⟩
    rcases hf x (s's hx) (min r 1) (lt_min rpos zero_lt_one) with ⟨R', hR'⟩
    exact
      ⟨R', ⟨hR'.1, hR'.2.1, hR'.2.2.trans_le (min_le_right _ _)⟩,
        Subset.trans (closedBall_subset_ball (hR'.2.2.trans_le (min_le_left _ _))) hr⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
    t0 : Set α
    r0 : α → Real
    t0_count : t0.Countable
    t0s : HasSubset.Subset t0 s
    hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
    μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
    t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
    s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
    s's : HasSubset.Subset s' s
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    v : Set α
    s'v : Superset v s'
    v_open : IsOpen v
    μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
    this : ∀ (x : α), Membership.mem s' x → Exists fun r1 => And (Membership.mem ( …
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  choose! r1 hr1 using this
  let q : BallPackage s' α :=
    { c := fun x => x
      r := fun x => r1 x
      rpos := fun x => (hr1 x.1 x.2).1.2.1
      r_bound := 1
      r_le := fun x => (hr1 x.1 x.2).1.2.2.le }
  -- by Besicovitch, we cover `s'` with at most `N` families of disjoint balls, all included in
  -- a suitable neighborhood `v` of `s'`.
  obtain ⟨S, S_disj, hS⟩ :
    ∃ S : Fin N → Set s',
      (∀ i : Fin N, (S i).PairwiseDisjoint fun j => closedBall (q.c j) (q.r j)) ∧
        range q.c ⊆ ⋃ i : Fin N, ⋃ j ∈ S i, ball (q.c j) (q.r j) :=
    exist_disjoint_covering_families hτ H q
  have S_count : ∀ i, (S i).Countable := by
    intro i
    apply (S_disj i).countable_of_nonempty_interior fun j _ => ?_
    have : (ball (j : α) (r1 j)).Nonempty := nonempty_ball.2 (q.rpos _)
    exact this.mono ball_subset_interior_closedBall
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
    t0 : Set α
    r0 : α → Real
    t0_count : t0.Countable
    t0s : HasSubset.Subset t0 s
    hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
    μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
    t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
    s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
    s's : HasSubset.Subset s' s
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    v : Set α
    s'v : Superset v s'
    v_open : IsOpen v
    μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
    r1 : α → Real
    hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
    q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
    S : Fin N → Set ↑s'
    S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
    hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
    S_count : ∀ (i : Fin N), (S i).Countable
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  let r x := if x ∈ s' then r1 x else r0 x
  have r_t0 : ∀ x ∈ t0, r x = r0 x := by
    intro x hx
    have : ¬x ∈ s' := by
      simp only [s', not_exists, exists_prop, mem_iUnion, mem_closedBall, not_and, not_lt, not_le,
        mem_diff, not_forall]
      intro _
      refine ⟨x, hx, ?_⟩
      rw [dist_self]
      exact (hr0 x hx).2.1.le
    simp only [r, if_neg this]
  -- the desired covering set is given by the union of the families constructed in the first and
  -- second steps.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝⁶ : MetricSpace α
    inst✝⁵ : SecondCountableTopology α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : OpensMeasurableSpace α
    inst✝² : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.SFinite μ
    inst✝ : μ.OuterRegular
    ε : ENNReal
    hε : Ne ε 0
    f : α → Set Real
    s : Set α
    hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
    u : Set α
    su : Superset u s
    u_open : IsOpen u
    μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
    R : α → Real
    hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
    t0 : Set α
    r0 : α → Real
    t0_count : t0.Countable
    t0s : HasSubset.Subset t0 s
    hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
    μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
    t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
    s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
    s's : HasSubset.Subset s' s
    N : Nat
    τ : Real
    hτ : LT.lt 1 τ
    H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
    v : Set α
    s'v : Superset v s'
    v_open : IsOpen v
    μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
    r1 : α → Real
    hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
    q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
    S : Fin N → Set ↑s'
    S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
    hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
    S_count : ∀ (i : Fin N), (S i).Countable
    r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
    r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
    ⊢ Exists fun t => Exists fun r => And t.Countable (And (HasSubset.Subset t s)  …
  -/
  refine ⟨t0 ∪ ⋃ i : Fin N, ((↑) : s' → α) '' S i, r, ?_, ?_, ?_, ?_, ?_⟩
  -- it remains to check that they have the desired properties
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      ⊢ (Union.union t0 (Set.iUnion fun i => Set.image Subtype.val (S i))).Countable
    -/
  · exact t0_count.union (countable_iUnion fun i => (S_count i).image _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      ⊢ HasSubset.Subset (Union.union t0 (Set.iUnion fun i => Set.image Subtype.val  …
    -/
  · simp only [t0s, true_and, union_subset_iff, image_subset_iff, iUnion_subset_iff]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      ⊢ ∀ (i : Fin N), HasSubset.Subset (S i) (Set.preimage Subtype.val s)
    -/
    intro i x _
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      i : Fin N
      x : Subtype fun x => Membership.mem s' x
      a✝ : Membership.mem (S i) x
      ⊢ Membership.mem (Set.preimage Subtype.val s) x
    -/
    exact s's x.2
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      ⊢ ∀ (x : α), Membership.mem (Union.union t0 (Set.iUnion fun i => Set.image Sub …
    -/
  · intro x hx
    cases hx with
    | inl hx =>
      rw [r_t0 x hx]
      exact (hr0 _ hx).1
    | inr hx =>
      have h'x : x ∈ s' := by
        simp only [mem_iUnion, mem_image] at hx
        rcases hx with ⟨i, y, _, rfl⟩
        exact y.2
      simp only [r, if_pos h'x, (hr1 x h'x).1.1]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      ⊢ HasSubset.Subset s (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBal …
    -/
  · intro x hx
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u_1
      inst✝⁶ : MetricSpace α
      inst✝⁵ : SecondCountableTopology α
      inst✝⁴ : MeasurableSpace α
      inst✝³ : OpensMeasurableSpace α
      inst✝² : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝¹ : MeasureTheory.SFinite μ
      inst✝ : μ.OuterRegular
      ε : ENNReal
      hε : Ne ε 0
      f : α → Set Real
      s : Set α
      hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
      u : Set α
      su : Superset u s
      u_open : IsOpen u
      μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
      R : α → Real
      hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
      t0 : Set α
      r0 : α → Real
      t0_count : t0.Countable
      t0s : HasSubset.Subset t0 s
      hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
      μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
      t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
      s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
      s's : HasSubset.Subset s' s
      N : Nat
      τ : Real
      hτ : LT.lt 1 τ
      H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
      v : Set α
      s'v : Superset v s'
      v_open : IsOpen v
      μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
      r1 : α → Real
      hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
      q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
      S : Fin N → Set ↑s'
      S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
      hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
      S_count : ∀ (i : Fin N), (S i).Countable
      r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
      r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
    -/
    by_cases h'x : x ∈ s'
    · obtain ⟨i, y, ySi, xy⟩ : ∃ (i : Fin N) (y : ↥s'), y ∈ S i ∧ x ∈ ball (y : α) (r1 y) := by
        have A : x ∈ range q.c := by
          simpa only [q, not_exists, exists_prop, mem_iUnion, mem_closedBall, not_and,
            not_le, mem_setOf_eq, Subtype.range_coe_subtype, mem_diff] using h'x
        simpa only [mem_iUnion, mem_image, bex_def] using hS A
      /-
        case pos.intro.intro.intro
        α : Type u_1
        inst✝⁶ : MetricSpace α
        inst✝⁵ : SecondCountableTopology α
        inst✝⁴ : MeasurableSpace α
        inst✝³ : OpensMeasurableSpace α
        inst✝² : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SFinite μ
        inst✝ : μ.OuterRegular
        ε : ENNReal
        hε : Ne ε 0
        f : α → Set Real
        s : Set α
        hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
        u : Set α
        su : Superset u s
        u_open : IsOpen u
        μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
        R : α → Real
        hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
        t0 : Set α
        r0 : α → Real
        t0_count : t0.Countable
        t0s : HasSubset.Subset t0 s
        hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
        μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
        t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
        s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
        s's : HasSubset.Subset s' s
        N : Nat
        τ : Real
        hτ : LT.lt 1 τ
        H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
        v : Set α
        s'v : Superset v s'
        v_open : IsOpen v
        μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
        r1 : α → Real
        hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
        q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
        S : Fin N → Set ↑s'
        S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
        hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
        S_count : ∀ (i : Fin N), (S i).Countable
        r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
        r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
        x : α
        hx : Membership.mem s x
        h'x : Membership.mem s' x
        i : Fin N
        y : ↑s'
        ySi : Membership.mem (S i) y
        xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
        ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
      -/
      refine mem_iUnion₂.2 ⟨y, Or.inr ?_, ?_⟩
        /-
          case pos.intro.intro.intro.refine_1
          α : Type u_1
          inst✝⁶ : MetricSpace α
          inst✝⁵ : SecondCountableTopology α
          inst✝⁴ : MeasurableSpace α
          inst✝³ : OpensMeasurableSpace α
          inst✝² : HasBesicovitchCovering α
          μ : MeasureTheory.Measure α
          inst✝¹ : MeasureTheory.SFinite μ
          inst✝ : μ.OuterRegular
          ε : ENNReal
          hε : Ne ε 0
          f : α → Set Real
          s : Set α
          hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
          u : Set α
          su : Superset u s
          u_open : IsOpen u
          μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
          R : α → Real
          hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
          t0 : Set α
          r0 : α → Real
          t0_count : t0.Countable
          t0s : HasSubset.Subset t0 s
          hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
          μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
          t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
          s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
          s's : HasSubset.Subset s' s
          N : Nat
          τ : Real
          hτ : LT.lt 1 τ
          H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
          v : Set α
          s'v : Superset v s'
          v_open : IsOpen v
          μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
          r1 : α → Real
          hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
          q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
          S : Fin N → Set ↑s'
          S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
          hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
          S_count : ∀ (i : Fin N), (S i).Countable
          r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
          r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
          x : α
          hx : Membership.mem s x
          h'x : Membership.mem s' x
          i : Fin N
          y : ↑s'
          ySi : Membership.mem (S i) y
          xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
          ⊢ Membership.mem (Set.iUnion fun i => Set.image Subtype.val (S i)) ↑y
        -/
      · simp only [mem_iUnion, mem_image]
        /-
          case pos.intro.intro.intro.refine_1
          α : Type u_1
          inst✝⁶ : MetricSpace α
          inst✝⁵ : SecondCountableTopology α
          inst✝⁴ : MeasurableSpace α
          inst✝³ : OpensMeasurableSpace α
          inst✝² : HasBesicovitchCovering α
          μ : MeasureTheory.Measure α
          inst✝¹ : MeasureTheory.SFinite μ
          inst✝ : μ.OuterRegular
          ε : ENNReal
          hε : Ne ε 0
          f : α → Set Real
          s : Set α
          hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
          u : Set α
          su : Superset u s
          u_open : IsOpen u
          μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
          R : α → Real
          hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
          t0 : Set α
          r0 : α → Real
          t0_count : t0.Countable
          t0s : HasSubset.Subset t0 s
          hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
          μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
          t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
          s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
          s's : HasSubset.Subset s' s
          N : Nat
          τ : Real
          hτ : LT.lt 1 τ
          H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
          v : Set α
          s'v : Superset v s'
          v_open : IsOpen v
          μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
          r1 : α → Real
          hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
          q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
          S : Fin N → Set ↑s'
          S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
          hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
          S_count : ∀ (i : Fin N), (S i).Countable
          r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
          r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
          x : α
          hx : Membership.mem s x
          h'x : Membership.mem s' x
          i : Fin N
          y : ↑s'
          ySi : Membership.mem (S i) y
          xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
          ⊢ Exists fun i => Exists fun x => And (Membership.mem (S i) x) (Eq ↑x ↑y)
        -/
        exact ⟨i, y, ySi, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.intro.intro.refine_2
          α : Type u_1
          inst✝⁶ : MetricSpace α
          inst✝⁵ : SecondCountableTopology α
          inst✝⁴ : MeasurableSpace α
          inst✝³ : OpensMeasurableSpace α
          inst✝² : HasBesicovitchCovering α
          μ : MeasureTheory.Measure α
          inst✝¹ : MeasureTheory.SFinite μ
          inst✝ : μ.OuterRegular
          ε : ENNReal
          hε : Ne ε 0
          f : α → Set Real
          s : Set α
          hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
          u : Set α
          su : Superset u s
          u_open : IsOpen u
          μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
          R : α → Real
          hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
          t0 : Set α
          r0 : α → Real
          t0_count : t0.Countable
          t0s : HasSubset.Subset t0 s
          hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
          μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
          t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
          s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
          s's : HasSubset.Subset s' s
          N : Nat
          τ : Real
          hτ : LT.lt 1 τ
          H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
          v : Set α
          s'v : Superset v s'
          v_open : IsOpen v
          μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
          r1 : α → Real
          hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
          q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
          S : Fin N → Set ↑s'
          S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
          hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
          S_count : ∀ (i : Fin N), (S i).Countable
          r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
          r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
          x : α
          hx : Membership.mem s x
          h'x : Membership.mem s' x
          i : Fin N
          y : ↑s'
          ySi : Membership.mem (S i) y
          xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
          ⊢ Membership.mem (Metric.closedBall (↑y) (r ↑y)) x
        -/
      · have : (y : α) ∈ s' := y.2
        /-
          case pos.intro.intro.intro.refine_2
          α : Type u_1
          inst✝⁶ : MetricSpace α
          inst✝⁵ : SecondCountableTopology α
          inst✝⁴ : MeasurableSpace α
          inst✝³ : OpensMeasurableSpace α
          inst✝² : HasBesicovitchCovering α
          μ : MeasureTheory.Measure α
          inst✝¹ : MeasureTheory.SFinite μ
          inst✝ : μ.OuterRegular
          ε : ENNReal
          hε : Ne ε 0
          f : α → Set Real
          s : Set α
          hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
          u : Set α
          su : Superset u s
          u_open : IsOpen u
          μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
          R : α → Real
          hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
          t0 : Set α
          r0 : α → Real
          t0_count : t0.Countable
          t0s : HasSubset.Subset t0 s
          hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
          μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
          t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
          s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
          s's : HasSubset.Subset s' s
          N : Nat
          τ : Real
          hτ : LT.lt 1 τ
          H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
          v : Set α
          s'v : Superset v s'
          v_open : IsOpen v
          μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
          r1 : α → Real
          hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
          q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
          S : Fin N → Set ↑s'
          S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
          hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
          S_count : ∀ (i : Fin N), (S i).Countable
          r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
          r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
          x : α
          hx : Membership.mem s x
          h'x : Membership.mem s' x
          i : Fin N
          y : ↑s'
          ySi : Membership.mem (S i) y
          xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
          this : Membership.mem s' ↑y
          ⊢ Membership.mem (Metric.closedBall (↑y) (r ↑y)) x
        -/
        simp only [r, if_pos this]
        /-
          case pos.intro.intro.intro.refine_2
          α : Type u_1
          inst✝⁶ : MetricSpace α
          inst✝⁵ : SecondCountableTopology α
          inst✝⁴ : MeasurableSpace α
          inst✝³ : OpensMeasurableSpace α
          inst✝² : HasBesicovitchCovering α
          μ : MeasureTheory.Measure α
          inst✝¹ : MeasureTheory.SFinite μ
          inst✝ : μ.OuterRegular
          ε : ENNReal
          hε : Ne ε 0
          f : α → Set Real
          s : Set α
          hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
          u : Set α
          su : Superset u s
          u_open : IsOpen u
          μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
          R : α → Real
          hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
          t0 : Set α
          r0 : α → Real
          t0_count : t0.Countable
          t0s : HasSubset.Subset t0 s
          hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
          μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
          t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
          s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
          s's : HasSubset.Subset s' s
          N : Nat
          τ : Real
          hτ : LT.lt 1 τ
          H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
          v : Set α
          s'v : Superset v s'
          v_open : IsOpen v
          μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
          r1 : α → Real
          hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
          q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
          S : Fin N → Set ↑s'
          S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
          hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
          S_count : ∀ (i : Fin N), (S i).Countable
          r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
          r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
          x : α
          hx : Membership.mem s x
          h'x : Membership.mem s' x
          i : Fin N
          y : ↑s'
          ySi : Membership.mem (S i) y
          xy : Membership.mem (Metric.ball (↑y) (r1 ↑y)) x
          this : Membership.mem s' ↑y
          ⊢ Membership.mem (Metric.closedBall (↑y) (r1 ↑y)) x
        -/
        exact ball_subset_closedBall xy
        /-
          🎉 no goals
        -/
    · obtain ⟨y, yt0, hxy⟩ : ∃ y : α, y ∈ t0 ∧ x ∈ closedBall y (r0 y) := by
        simpa [s', hx, -mem_closedBall] using h'x
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝⁶ : MetricSpace α
        inst✝⁵ : SecondCountableTopology α
        inst✝⁴ : MeasurableSpace α
        inst✝³ : OpensMeasurableSpace α
        inst✝² : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SFinite μ
        inst✝ : μ.OuterRegular
        ε : ENNReal
        hε : Ne ε 0
        f : α → Set Real
        s : Set α
        hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
        u : Set α
        su : Superset u s
        u_open : IsOpen u
        μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
        R : α → Real
        hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
        t0 : Set α
        r0 : α → Real
        t0_count : t0.Countable
        t0s : HasSubset.Subset t0 s
        hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
        μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
        t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
        s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
        s's : HasSubset.Subset s' s
        N : Nat
        τ : Real
        hτ : LT.lt 1 τ
        H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
        v : Set α
        s'v : Superset v s'
        v_open : IsOpen v
        μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
        r1 : α → Real
        hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
        q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
        S : Fin N → Set ↑s'
        S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
        hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
        S_count : ∀ (i : Fin N), (S i).Countable
        r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
        r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
        x : α
        hx : Membership.mem s x
        h'x : Not (Membership.mem s' x)
        y : α
        yt0 : Membership.mem t0 y
        hxy : Membership.mem (Metric.closedBall y (r0 y)) x
        ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.closedBall x  …
      -/
      refine mem_iUnion₂.2 ⟨y, Or.inl yt0, ?_⟩
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝⁶ : MetricSpace α
        inst✝⁵ : SecondCountableTopology α
        inst✝⁴ : MeasurableSpace α
        inst✝³ : OpensMeasurableSpace α
        inst✝² : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.SFinite μ
        inst✝ : μ.OuterRegular
        ε : ENNReal
        hε : Ne ε 0
        f : α → Set Real
        s : Set α
        hf : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (f …
        u : Set α
        su : Superset u s
        u_open : IsOpen u
        μu : LE.le (μ u) (HAdd.hAdd (μ s) (HDiv.hDiv ε 2))
        R : α → Real
        hR : ∀ (x : α), Membership.mem s x → And (GT.gt (R x) 0) (HasSubset.Subset (Me …
        t0 : Set α
        r0 : α → Real
        t0_count : t0.Countable
        t0s : HasSubset.Subset t0 s
        hr0 : ∀ (x : α), Membership.mem t0 x → Membership.mem (Inter.inter (f x) (Set. …
        μt0 : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clo …
        t0_disj : t0.PairwiseDisjoint fun x => Metric.closedBall x (r0 x)
        s' : Set α := SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.cl …
        s's : HasSubset.Subset s' s
        N : Nat
        τ : Real
        hτ : LT.lt 1 τ
        H : IsEmpty (Besicovitch.SatelliteConfig α N τ)
        v : Set α
        s'v : Superset v s'
        v_open : IsOpen v
        μv : LE.le (μ v) (HAdd.hAdd (μ s') (HDiv.hDiv (HDiv.hDiv ε 2) ↑N))
        r1 : α → Real
        hr1 : ∀ (x : α), Membership.mem s' x → And (Membership.mem (Inter.inter (f x)  …
        q : Besicovitch.BallPackage (↑s') α := { c := fun x => ↑x, r := fun x => r1 ↑x …
        S : Fin N → Set ↑s'
        S_disj : ∀ (i : Fin N), (S i).PairwiseDisjoint fun j => Metric.closedBall (q.c …
        hS : HasSubset.Subset (Set.range q.c) (Set.iUnion fun i => Set.iUnion fun j => …
        S_count : ∀ (i : Fin N), (S i).Countable
        r : α → Real := fun x => ite (Membership.mem s' x) (r1 x) (r0 x)
        r_t0 : ∀ (x : α), Membership.mem t0 x → Eq (r x) (r0 x)
        x : α
        hx : Membership.mem s x
        h'x : Not (Membership.mem s' x)
        y : α
        yt0 : Membership.mem t0 y
        hxy : Membership.mem (Metric.closedBall y (r0 y)) x
        ⊢ Membership.mem (Metric.closedBall y (r y)) x
      -/
      rwa [r_t0 _ yt0]
      /-
        🎉 no goals
      -/
  -- the only nontrivial property is the measure control, which we check now
  · -- the sets in the first step have measure at most `μ s + ε / 2`
    have A : (∑' x : t0, μ (closedBall x (r x))) ≤ μ s + ε / 2 :=
      calc
        (∑' x : t0, μ (closedBall x (r x))) = ∑' x : t0, μ (closedBall x (r0 x)) := by
          congr 1; ext x; rw [r_t0 x x.2]
        _ = μ (⋃ x : t0, closedBall x (r0 x)) := by
          haveI : Encodable t0 := t0_count.toEncodable
          rw [measure_iUnion]
          · exact (pairwise_subtype_iff_pairwise_set _ _).2 t0_disj
          · exact fun i => measurableSet_closedBall
        _ ≤ μ u := by
          apply measure_mono
          simp only [SetCoe.forall, Subtype.coe_mk, iUnion_subset_iff]
          intro x hx
          apply Subset.trans (closedBall_subset_ball (hr0 x hx).2.2) (hR x (t0s hx)).2
        _ ≤ μ s + ε / 2 := μu
    -- each subfamily in the second step has measure at most `ε / (2 N)`.
    have B : ∀ i : Fin N, (∑' x : ((↑) : s' → α) '' S i, μ (closedBall x (r x))) ≤ ε / 2 / N :=
      fun i =>
      calc
        (∑' x : ((↑) : s' → α) '' S i, μ (closedBall x (r x))) =
            ∑' x : S i, μ (closedBall x (r x)) := by
          have : InjOn ((↑) : s' → α) (S i) := Subtype.val_injective.injOn
          let F : S i ≃ ((↑) : s' → α) '' S i := this.bijOn_image.equiv _
          exact (F.tsum_eq fun x => μ (closedBall x (r x))).symm
        _ = ∑' x : S i, μ (closedBall x (r1 x)) := by
          congr 1; ext x; have : (x : α) ∈ s' := x.1.2; simp only [s', r, if_pos this]
        _ = μ (⋃ x : S i, closedBall x (r1 x)) := by
          haveI : Encodable (S i) := (S_count i).toEncodable
          rw [measure_iUnion]
          · exact (pairwise_subtype_iff_pairwise_set _ _).2 (S_disj i)
          · exact fun i => measurableSet_closedBall
        _ ≤ μ v := by
          apply measure_mono
          simp only [SetCoe.forall, Subtype.coe_mk, iUnion_subset_iff]
          intro x xs' _
          exact (hr1 x xs').2
        _ ≤ ε / 2 / N := by have : μ s' = 0 := μt0; rwa [this, zero_add] at μv
    -- add up all these to prove the desired estimate
    calc
      (∑' x : ↥(t0 ∪ ⋃ i : Fin N, ((↑) : s' → α) '' S i), μ (closedBall x (r x))) ≤
          (∑' x : t0, μ (closedBall x (r x))) +
            ∑' x : ⋃ i : Fin N, ((↑) : s' → α) '' S i, μ (closedBall x (r x)) :=
        ENNReal.tsum_union_le (fun x => μ (closedBall x (r x))) _ _
      _ ≤
          (∑' x : t0, μ (closedBall x (r x))) +
            ∑ i : Fin N, ∑' x : ((↑) : s' → α) '' S i, μ (closedBall x (r x)) :=
        (add_le_add le_rfl (ENNReal.tsum_iUnion_le (fun x => μ (closedBall x (r x))) _))
      _ ≤ μ s + ε / 2 + ∑ i : Fin N, ε / 2 / N := by
        gcongr
        apply B
      _ ≤ μ s + ε / 2 + ε / 2 := by
        gcongr
        simp only [Finset.card_fin, Finset.sum_const, nsmul_eq_mul, ENNReal.mul_div_le]
      _ = μ s + ε := by rw [add_assoc, ENNReal.add_halves]


/-- In a space with the Besicovitch covering property, the set of closed balls with positive radius
forms a Vitali family. This is essentially a restatement of the measurable Besicovitch theorem. -/
protected def vitaliFamily (μ : Measure α) [SFinite μ] : VitaliFamily μ where
  setsAt x := (fun r : ℝ => closedBall x r) '' Ioi (0 : ℝ)
  measurableSet _ := forall_mem_image.2 fun _ _ ↦ isClosed_ball.measurableSet
  nonempty_interior _ := forall_mem_image.2 fun _ rpos ↦
    (nonempty_ball.2 rpos).mono ball_subset_interior_closedBall
  nontrivial x ε εpos := ⟨closedBall x ε, mem_image_of_mem _ εpos, Subset.rfl⟩
  covering := by
    /-
      α : Type u_1
      inst✝⁵ : MetricSpace α
      β : Type u
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      ⊢ ∀ (s : Set α) (f : α → Set (Set α)), (∀ (x : α), Membership.mem s x → HasSub …
    -/
    intro s f fsubset ffine
    /-
      α : Type u_1
      inst✝⁵ : MetricSpace α
      β : Type u
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      s : Set α
      f : α → Set (Set α)
      fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
      ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      ⊢ Exists fun t => And (∀ (p : Prod α (Set α)), Membership.mem t p → Membership …
    -/
    let g : α → Set ℝ := fun x => {r | 0 < r ∧ closedBall x r ∈ f x}
    have A : ∀ x ∈ s, ∀ δ > 0, (g x ∩ Ioo 0 δ).Nonempty := by
      intro x xs δ δpos
      obtain ⟨t, tf, ht⟩ : ∃ (t : Set α), t ∈ f x ∧ t ⊆ closedBall x (δ / 2) :=
        ffine x xs (δ / 2) (half_pos δpos)
      obtain ⟨r, rpos, rfl⟩ : ∃ r : ℝ, 0 < r ∧ closedBall x r = t := by simpa using fsubset x xs tf
      rcases le_total r (δ / 2) with (H | H)
      · exact ⟨r, ⟨rpos, tf⟩, ⟨rpos, H.trans_lt (half_lt_self δpos)⟩⟩
      · have : closedBall x r = closedBall x (δ / 2) :=
          Subset.antisymm ht (closedBall_subset_closedBall H)
        rw [this] at tf
        exact ⟨δ / 2, ⟨half_pos δpos, tf⟩, ⟨half_pos δpos, half_lt_self δpos⟩⟩
    obtain ⟨t, r, _, ts, tg, μt, tdisj⟩ :
      ∃ (t : Set α) (r : α → ℝ),
        t.Countable ∧
          t ⊆ s ∧
            (∀ x ∈ t, r x ∈ g x ∩ Ioo 0 1) ∧
              μ (s \ ⋃ x ∈ t, closedBall x (r x)) = 0 ∧
                t.PairwiseDisjoint fun x => closedBall x (r x) :=
      exists_disjoint_closedBall_covering_ae μ g s A (fun _ => 1) fun _ _ => zero_lt_one
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝⁵ : MetricSpace α
      β : Type u
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      s : Set α
      f : α → Set (Set α)
      fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
      ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
      A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
      t : Set α
      r : α → Real
      left✝ : t.Countable
      ts : HasSubset.Subset t s
      tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
      μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
      tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
      ⊢ Exists fun t => And (∀ (p : Prod α (Set α)), Membership.mem t p → Membership …
    -/
    let F : α → α × Set α := fun x => (x, closedBall x (r x))
    /-
      case intro.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝⁵ : MetricSpace α
      β : Type u
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      s : Set α
      f : α → Set (Set α)
      fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
      ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
      A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
      t : Set α
      r : α → Real
      left✝ : t.Countable
      ts : HasSubset.Subset t s
      tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
      μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
      tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
      F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
      ⊢ Exists fun t => And (∀ (p : Prod α (Set α)), Membership.mem t p → Membership …
    -/
    refine ⟨F '' t, ?_, ?_, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        inst✝⁵ : MetricSpace α
        β : Type u
        inst✝⁴ : SecondCountableTopology α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
        A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
        t : Set α
        r : α → Real
        left✝ : t.Countable
        ts : HasSubset.Subset t s
        tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
        μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
        tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
        F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
        ⊢ ∀ (p : Prod α (Set α)), Membership.mem (Set.image F t) p → Membership.mem s  …
      -/
    · rintro - ⟨x, hx, rfl⟩; exact ts hx
                             /-
                               🎉 no goals
                             -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        inst✝⁵ : MetricSpace α
        β : Type u
        inst✝⁴ : SecondCountableTopology α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
        A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
        t : Set α
        r : α → Real
        left✝ : t.Countable
        ts : HasSubset.Subset t s
        tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
        μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
        tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
        F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
        ⊢ (Set.image F t).PairwiseDisjoint fun p => p.2
      -/
    · rintro p ⟨x, hx, rfl⟩ q ⟨y, hy, rfl⟩ hxy
      /-
        case intro.intro.intro.intro.intro.intro.refine_2.intro.intro.intro.intro
        α : Type u_1
        inst✝⁵ : MetricSpace α
        β : Type u
        inst✝⁴ : SecondCountableTopology α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
        A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
        t : Set α
        r : α → Real
        left✝ : t.Countable
        ts : HasSubset.Subset t s
        tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
        μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
        tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
        F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
        x : α
        hx : Membership.mem t x
        y : α
        hy : Membership.mem t y
        hxy : Ne (F x) (F y)
        ⊢ Function.onFun Disjoint (fun p => p.2) (F x) (F y)
      -/
      exact tdisj hx hy (ne_of_apply_ne F hxy)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_3
        α : Type u_1
        inst✝⁵ : MetricSpace α
        β : Type u
        inst✝⁴ : SecondCountableTopology α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
        A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
        t : Set α
        r : α → Real
        left✝ : t.Countable
        ts : HasSubset.Subset t s
        tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
        μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
        tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
        F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
        ⊢ ∀ (p : Prod α (Set α)), Membership.mem (Set.image F t) p → Membership.mem (f …
      -/
    · rintro - ⟨x, hx, rfl⟩; exact (tg x hx).1.2
                             /-
                               🎉 no goals
                             -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_4
        α : Type u_1
        inst✝⁵ : MetricSpace α
        β : Type u
        inst✝⁴ : SecondCountableTopology α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : HasBesicovitchCovering α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SFinite μ
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => Se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        g : α → Set Real := fun x => setOf fun r => And (LT.lt 0 r) (Membership.mem (f …
        A : ∀ (x : α), Membership.mem s x → ∀ (δ : Real), GT.gt δ 0 → (Inter.inter (g  …
        t : Set α
        r : α → Real
        left✝ : t.Countable
        ts : HasSubset.Subset t s
        tg : ∀ (x : α), Membership.mem t x → Membership.mem (Inter.inter (g x) (Set.Io …
        μt : Eq (μ (SDiff.sdiff s (Set.iUnion fun x => Set.iUnion fun h => Metric.clos …
        tdisj : t.PairwiseDisjoint fun x => Metric.closedBall x (r x)
        F : α → Prod α (Set α) := fun x => { fst := x, snd := Metric.closedBall x (r x …
        ⊢ Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun h => p.2))) 0
      -/
    · rwa [biUnion_image]
      /-
        🎉 no goals
      -/


/-- The main feature of the Besicovitch Vitali family is that its filter at a point `x` corresponds
to convergence along closed balls. We record one of the two implications here, which will enable us
to deduce specific statements on differentiation of measures in this context from the general
versions. -/
theorem tendsto_filterAt (μ : Measure α) [SFinite μ] (x : α) :
    Tendsto (fun r => closedBall x r) (𝓝[>] 0) ((Besicovitch.vitaliFamily μ).filterAt x) := by
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    x : α
    ⊢ Filter.Tendsto (fun r => Metric.closedBall x r) (nhdsWithin 0 (Set.Ioi 0)) ( …
  -/
  intro s hs
  /-
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    x : α
    s : Set (Set α)
    hs : Membership.mem ((Besicovitch.vitaliFamily μ).filterAt x) s
    ⊢ Membership.mem (Filter.map (fun r => Metric.closedBall x r) (nhdsWithin 0 (S …
  -/
  simp only [mem_map]
  obtain ⟨ε, εpos, hε⟩ :
    ∃ (ε : ℝ), ε > 0 ∧
      ∀ a : Set α, a ∈ (Besicovitch.vitaliFamily μ).setsAt x → a ⊆ closedBall x ε → a ∈ s :=
    (VitaliFamily.mem_filterAt_iff _).1 hs
  /-
    case intro.intro
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    x : α
    s : Set (Set α)
    hs : Membership.mem ((Besicovitch.vitaliFamily μ).filterAt x) s
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ (a : Set α), Membership.mem ((Besicovitch.vitaliFamily μ).setsAt x) a → …
    ⊢ Membership.mem (nhdsWithin 0 (Set.Ioi 0)) (Set.preimage (fun r => Metric.clo …
  -/
  filter_upwards [Ioc_mem_nhdsGT εpos] with _r hr
  /-
    case h
    α : Type u_1
    inst✝⁵ : MetricSpace α
    inst✝⁴ : SecondCountableTopology α
    inst✝³ : MeasurableSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : HasBesicovitchCovering α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    x : α
    s : Set (Set α)
    hs : Membership.mem ((Besicovitch.vitaliFamily μ).filterAt x) s
    ε : Real
    εpos : GT.gt ε 0
    hε : ∀ (a : Set α), Membership.mem ((Besicovitch.vitaliFamily μ).setsAt x) a → …
    _r : Real
    hr : Membership.mem (Set.Ioc 0 ε) _r
    ⊢ Membership.mem (Set.preimage (fun r => Metric.closedBall x r) s) _r
  -/
  apply hε
    /-
      case h.a
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      x : α
      s : Set (Set α)
      hs : Membership.mem ((Besicovitch.vitaliFamily μ).filterAt x) s
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (a : Set α), Membership.mem ((Besicovitch.vitaliFamily μ).setsAt x) a → …
      _r : Real
      hr : Membership.mem (Set.Ioc 0 ε) _r
      ⊢ Membership.mem ((Besicovitch.vitaliFamily μ).setsAt x) ((fun r => Metric.clo …
    -/
  · exact mem_image_of_mem _ hr.1
    /-
      🎉 no goals
    -/
    /-
      case h.a
      α : Type u_1
      inst✝⁵ : MetricSpace α
      inst✝⁴ : SecondCountableTopology α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : HasBesicovitchCovering α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      x : α
      s : Set (Set α)
      hs : Membership.mem ((Besicovitch.vitaliFamily μ).filterAt x) s
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ (a : Set α), Membership.mem ((Besicovitch.vitaliFamily μ).setsAt x) a → …
      _r : Real
      hr : Membership.mem (Set.Ioc 0 ε) _r
      ⊢ HasSubset.Subset ((fun r => Metric.closedBall x r) _r) (Metric.closedBall x ε)
    -/
  · exact closedBall_subset_closedBall hr.2
    /-
      🎉 no goals
    -/


/-- In a space with the Besicovitch covering property, the ratio of the measure of balls converges
almost surely to the Radon-Nikodym derivative. -/
theorem ae_tendsto_rnDeriv (ρ μ : Measure β) [IsLocallyFiniteMeasure μ] [IsLocallyFiniteMeasure ρ] :
    ∀ᵐ x ∂μ,
      Tendsto (fun r => ρ (closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 (ρ.rnDeriv μ x)) := by
  /-
    β : Type u
    inst✝⁶ : MetricSpace β
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : BorelSpace β
    inst✝³ : SecondCountableTopology β
    inst✝² : HasBesicovitchCovering β
    ρ μ : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun r => HDiv.hDiv (ρ (Metric.cl …
  -/
  filter_upwards [VitaliFamily.ae_tendsto_rnDeriv (Besicovitch.vitaliFamily μ) ρ] with x hx
  /-
    case h
    β : Type u
    inst✝⁶ : MetricSpace β
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : BorelSpace β
    inst✝³ : SecondCountableTopology β
    inst✝² : HasBesicovitchCovering β
    ρ μ : MeasureTheory.Measure β
    inst✝¹ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure ρ
    x : β
    hx : Filter.Tendsto (fun a => HDiv.hDiv (ρ a) (μ a)) ((Besicovitch.vitaliFamil …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (ρ (Metric.closedBall x r)) (μ (Metric.cl …
  -/
  exact hx.comp (tendsto_filterAt μ x)
  /-
    🎉 no goals
  -/


/-- Given a measurable set `s`, then `μ (s ∩ closedBall x r) / μ (closedBall x r)` converges when
`r` tends to `0`, for almost every `x`. The limit is `1` for `x ∈ s` and `0` for `x ∉ s`.
This shows that almost every point of `s` is a Lebesgue density point for `s`.
A version for non-measurable sets holds, but it only gives the first conclusion,
see `ae_tendsto_measure_inter_div`. -/
theorem ae_tendsto_measure_inter_div_of_measurableSet (μ : Measure β) [IsLocallyFiniteMeasure μ]
    {s : Set β} (hs : MeasurableSet s) :
    ∀ᵐ x ∂μ,
      Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0)
        (𝓝 (s.indicator 1 x)) := by
  filter_upwards [VitaliFamily.ae_tendsto_measure_inter_div_of_measurableSet
      (Besicovitch.vitaliFamily μ) hs]
  /-
    case h
    β : Type u
    inst✝⁵ : MetricSpace β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : BorelSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : HasBesicovitchCovering β
    μ : MeasureTheory.Measure β
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set β
    hs : MeasurableSet s
    ⊢ ∀ (a : β), Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter s a)) (μ a)) ( …
  -/
  intro x hx
  /-
    case h
    β : Type u
    inst✝⁵ : MetricSpace β
    inst✝⁴ : MeasurableSpace β
    inst✝³ : BorelSpace β
    inst✝² : SecondCountableTopology β
    inst✝¹ : HasBesicovitchCovering β
    μ : MeasureTheory.Measure β
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    s : Set β
    hs : MeasurableSet s
    x : β
    hx : Filter.Tendsto (fun a => HDiv.hDiv (μ (Inter.inter s a)) (μ a)) ((Besicov …
    ⊢ Filter.Tendsto (fun r => HDiv.hDiv (μ (Inter.inter s (Metric.closedBall x r) …
  -/
  exact hx.comp (tendsto_filterAt μ x)
  /-
    🎉 no goals
  -/


/-- Given an arbitrary set `s`, then `μ (s ∩ closedBall x r) / μ (closedBall x r)` converges
to `1` when `r` tends to `0`, for almost every `x` in `s`.
This shows that almost every point of `s` is a Lebesgue density point for `s`.
A stronger version holds for measurable sets, see `ae_tendsto_measure_inter_div_of_measurableSet`.

See also `IsUnifLocDoublingMeasure.ae_tendsto_measure_inter_div`. -/
theorem ae_tendsto_measure_inter_div (μ : Measure β) [IsLocallyFiniteMeasure μ] (s : Set β) :
    ∀ᵐ x ∂μ.restrict s,
      Tendsto (fun r => μ (s ∩ closedBall x r) / μ (closedBall x r)) (𝓝[>] 0) (𝓝 1) := by
  filter_upwards [VitaliFamily.ae_tendsto_measure_inter_div (Besicovitch.vitaliFamily μ) s] with x
    hx using hx.comp (tendsto_filterAt μ x)


