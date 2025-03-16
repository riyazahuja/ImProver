/-- `Covariant` is useful to formulate succinctly statements about the interactions between an
action of a Type on another one and a relation on the acted-upon Type.

See the `CovariantClass` doc-string for its meaning. -/
def Covariant : Prop :=
  ∀ (m) {n₁ n₂}, r n₁ n₂ → r (μ m n₁) (μ m n₂)


/-- `Contravariant` is useful to formulate succinctly statements about the interactions between an
action of a Type on another one and a relation on the acted-upon Type.

See the `ContravariantClass` doc-string for its meaning. -/
def Contravariant : Prop :=
  ∀ (m) {n₁ n₂}, r (μ m n₁) (μ m n₂) → r n₁ n₂


/-- Given an action `μ` of a Type `M` on a Type `N` and a relation `r` on `N`, informally, the
`CovariantClass` says that "the action `μ` preserves the relation `r`."

More precisely, the `CovariantClass` is a class taking two Types `M N`, together with an "action"
`μ : M → N → N` and a relation `r : N → N → Prop`.  Its unique field `elim` is the assertion that
for all `m ∈ M` and all elements `n₁, n₂ ∈ N`, if the relation `r` holds for the pair
`(n₁, n₂)`, then, the relation `r` also holds for the pair `(μ m n₁, μ m n₂)`,
obtained from `(n₁, n₂)` by acting upon it by `m`.

If `m : M` and `h : r n₁ n₂`, then `CovariantClass.elim m h : r (μ m n₁) (μ m n₂)`.
-/
class CovariantClass : Prop where
  /-- For all `m ∈ M` and all elements `n₁, n₂ ∈ N`, if the relation `r` holds for the pair
  `(n₁, n₂)`, then, the relation `r` also holds for the pair `(μ m n₁, μ m n₂)` -/
  protected elim : Covariant M N μ r


/-- Given an action `μ` of a Type `M` on a Type `N` and a relation `r` on `N`, informally, the
`ContravariantClass` says that "if the result of the action `μ` on a pair satisfies the
relation `r`, then the initial pair satisfied the relation `r`."

More precisely, the `ContravariantClass` is a class taking two Types `M N`, together with an
"action" `μ : M → N → N` and a relation `r : N → N → Prop`.  Its unique field `elim` is the
assertion that for all `m ∈ M` and all elements `n₁, n₂ ∈ N`, if the relation `r` holds for the
pair `(μ m n₁, μ m n₂)` obtained from `(n₁, n₂)` by acting upon it by `m`, then, the relation
`r` also holds for the pair `(n₁, n₂)`.

If `m : M` and `h : r (μ m n₁) (μ m n₂)`, then `ContravariantClass.elim m h : r n₁ n₂`.
-/
class ContravariantClass : Prop where
  /-- For all `m ∈ M` and all elements `n₁, n₂ ∈ N`, if the relation `r` holds for the
  pair `(μ m n₁, μ m n₂)` obtained from `(n₁, n₂)` by acting upon it by `m`, then, the relation
  `r` also holds for the pair `(n₁, n₂)`. -/
  protected elim : Contravariant M N μ r


/-- Typeclass for monotonicity of multiplication on the left,
namely `b₁ ≤ b₂ → a * b₁ ≤ a * b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommMonoid`. -/
abbrev MulLeftMono [Mul M] [LE M] : Prop :=
  CovariantClass M M (· * ·) (· ≤ ·)


/-- Typeclass for monotonicity of multiplication on the right,
namely `a₁ ≤ a₂ → a₁ * b ≤ a₂ * b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommMonoid`. -/
abbrev MulRightMono [Mul M] [LE M] : Prop :=
  CovariantClass M M (swap (· * ·)) (· ≤ ·)


/-- Typeclass for monotonicity of addition on the left,
namely `b₁ ≤ b₂ → a + b₁ ≤ a + b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommMonoid`. -/
abbrev AddLeftMono [Add M] [LE M] : Prop :=
  CovariantClass M M (· + ·) (· ≤ ·)


/-- Typeclass for monotonicity of addition on the right,
namely `a₁ ≤ a₂ → a₁ + b ≤ a₂ + b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommMonoid`. -/
abbrev AddRightMono [Add M] [LE M] : Prop :=
  CovariantClass M M (swap (· + ·)) (· ≤ ·)


attribute [to_additive existing] MulLeftMono MulRightMono


/-- Typeclass for monotonicity of multiplication on the left,
namely `b₁ < b₂ → a * b₁ < a * b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommGroup`. -/
abbrev MulLeftStrictMono [Mul M] [LT M] : Prop :=
  CovariantClass M M (· * ·) (· < ·)


/-- Typeclass for monotonicity of multiplication on the right,
namely `a₁ < a₂ → a₁ * b < a₂ * b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommGroup`. -/
abbrev MulRightStrictMono [Mul M] [LT M] : Prop :=
  CovariantClass M M (swap (· * ·)) (· < ·)


/-- Typeclass for monotonicity of addition on the left,
namely `b₁ < b₂ → a + b₁ < a + b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommGroup`. -/
abbrev AddLeftStrictMono [Add M] [LT M] : Prop :=
  CovariantClass M M (· + ·) (· < ·)


/-- Typeclass for monotonicity of addition on the right,
namely `a₁ < a₂ → a₁ + b < a₂ + b`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommGroup`. -/
abbrev AddRightStrictMono [Add M] [LT M] : Prop :=
  CovariantClass M M (swap (· + ·)) (· < ·)


attribute [to_additive existing] MulLeftStrictMono MulRightStrictMono


/-- Typeclass for strict reverse monotonicity of multiplication on the left,
namely `a * b₁ < a * b₂ → b₁ < b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommGroup`. -/
abbrev MulLeftReflectLT [Mul M] [LT M] : Prop :=
  ContravariantClass M M (· * ·) (· < ·)


/-- Typeclass for strict reverse monotonicity of multiplication on the right,
namely `a₁ * b < a₂ * b → a₁ < a₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCommGroup`. -/
abbrev MulRightReflectLT [Mul M] [LT M] : Prop :=
  ContravariantClass M M (swap (· * ·)) (· < ·)


/-- Typeclass for strict reverse monotonicity of addition on the left,
namely `a + b₁ < a + b₂ → b₁ < b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommGroup`. -/
abbrev AddLeftReflectLT [Add M] [LT M] : Prop :=
  ContravariantClass M M (· + ·) (· < ·)


/-- Typeclass for strict reverse monotonicity of addition on the right,
namely `a₁ * b < a₂ * b → a₁ < a₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedAddCommGroup`. -/
abbrev AddRightReflectLT [Add M] [LT M] : Prop :=
  ContravariantClass M M (swap (· + ·)) (· < ·)


attribute [to_additive existing] MulLeftReflectLT MulRightReflectLT


/-- Typeclass for reverse monotonicity of multiplication on the left,
namely `a * b₁ ≤ a * b₂ → b₁ ≤ b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCancelCommMonoid`. -/
abbrev MulLeftReflectLE [Mul M] [LE M] : Prop :=
  ContravariantClass M M (· * ·) (· ≤ ·)


/-- Typeclass for reverse monotonicity of multiplication on the right,
namely `a₁ * b ≤ a₂ * b → a₁ ≤ a₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCancelCommMonoid`. -/
abbrev MulRightReflectLE [Mul M] [LE M] : Prop :=
  ContravariantClass M M (swap (· * ·)) (· ≤ ·)


/-- Typeclass for reverse monotonicity of addition on the left,
namely `a + b₁ ≤ a + b₂ → b₁ ≤ b₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCancelAddCommMonoid`. -/
abbrev AddLeftReflectLE [Add M] [LE M] : Prop :=
  ContravariantClass M M (· + ·) (· ≤ ·)


/-- Typeclass for reverse monotonicity of addition on the right,
namely `a₁ + b ≤ a₂ + b → a₁ ≤ a₂`.

You should usually not use this very granular typeclass directly, but rather a typeclass like
`OrderedCancelAddCommMonoid`. -/
abbrev AddRightReflectLE [Add M] [LE M] : Prop :=
  ContravariantClass M M (swap (· + ·)) (· ≤ ·)


attribute [to_additive existing] MulLeftReflectLE MulRightReflectLE


theorem rel_iff_cov [CovariantClass M N μ r] [ContravariantClass M N μ r] (m : M) {a b : N} :
    r (μ m a) (μ m b) ↔ r a b :=
  ⟨ContravariantClass.elim _, CovariantClass.elim _⟩


theorem Covariant.flip (h : Covariant M N μ r) : Covariant M N μ (flip r) :=
  fun a _ _ ↦ h a


theorem Contravariant.flip (h : Contravariant M N μ r) : Contravariant M N μ (flip r) :=
  fun a _ _ ↦ h a


theorem act_rel_act_of_rel (m : M) {a b : N} (ab : r a b) : r (μ m a) (μ m b) :=
  CovariantClass.elim _ ab


@[to_additive]
theorem Group.covariant_iff_contravariant [Group N] :
    Covariant N N (· * ·) r ↔ Contravariant N N (· * ·) r := by
  /-
    N : Type u_2
    r : N → N → Prop
    inst✝ : Group N
    ⊢ Iff (Covariant N N (fun x1 x2 => HMul.hMul x1 x2) r) (Contravariant N N (fun …
  -/
  refine ⟨fun h a b c bc ↦ ?_, fun h a b c bc ↦ ?_⟩
    /-
      case refine_1
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Covariant N N (fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r ((fun x1 x2 => HMul.hMul x1 x2) a b) ((fun x1 x2 => HMul.hMul x1 x2) a c)
      ⊢ r b c
    -/
  · rw [← inv_mul_cancel_left a b, ← inv_mul_cancel_left a c]
    /-
      case refine_1
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Covariant N N (fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r ((fun x1 x2 => HMul.hMul x1 x2) a b) ((fun x1 x2 => HMul.hMul x1 x2) a c)
      ⊢ r (HMul.hMul (Inv.inv a) (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (HMul.hMul  …
    -/
    exact h a⁻¹ bc
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Contravariant N N (fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r b c
      ⊢ r ((fun x1 x2 => HMul.hMul x1 x2) a b) ((fun x1 x2 => HMul.hMul x1 x2) a c)
    -/
  · rw [← inv_mul_cancel_left a b, ← inv_mul_cancel_left a c] at bc
    /-
      case refine_2
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Contravariant N N (fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r (HMul.hMul (Inv.inv a) (HMul.hMul a b)) (HMul.hMul (Inv.inv a) (HMul.hM …
      ⊢ r ((fun x1 x2 => HMul.hMul x1 x2) a b) ((fun x1 x2 => HMul.hMul x1 x2) a c)
    -/
    exact h a⁻¹ bc
    /-
      🎉 no goals
    -/


@[to_additive]
instance (priority := 100) Group.covconv [Group N] [CovariantClass N N (· * ·) r] :
    ContravariantClass N N (· * ·) r :=
  ⟨Group.covariant_iff_contravariant.mp CovariantClass.elim⟩


@[to_additive]
theorem Group.mulLeftReflectLE_of_mulLeftMono [Group N] [LE N]
    [MulLeftMono N] : MulLeftReflectLE N :=
  inferInstance


@[to_additive]
theorem Group.mulLeftReflectLT_of_mulLeftStrictMono [Group N] [LT N]
    [MulLeftStrictMono N] : MulLeftReflectLT N :=
  inferInstance


@[to_additive]
theorem Group.covariant_swap_iff_contravariant_swap [Group N] :
    Covariant N N (swap (· * ·)) r ↔ Contravariant N N (swap (· * ·)) r := by
  /-
    N : Type u_2
    r : N → N → Prop
    inst✝ : Group N
    ⊢ Iff (Covariant N N (Function.swap fun x1 x2 => HMul.hMul x1 x2) r) (Contrava …
  -/
  refine ⟨fun h a b c bc ↦ ?_, fun h a b c bc ↦ ?_⟩
    /-
      case refine_1
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Covariant N N (Function.swap fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a b) (Function.swap (fun  …
      ⊢ r b c
    -/
  · rw [← mul_inv_cancel_right b a, ← mul_inv_cancel_right c a]
    /-
      case refine_1
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Covariant N N (Function.swap fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a b) (Function.swap (fun  …
      ⊢ r (HMul.hMul (HMul.hMul b a) (Inv.inv a)) (HMul.hMul (HMul.hMul c a) (Inv.in …
    -/
    exact h a⁻¹ bc
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Contravariant N N (Function.swap fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r b c
      ⊢ r (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a b) (Function.swap (fun x1  …
    -/
  · rw [← mul_inv_cancel_right b a, ← mul_inv_cancel_right c a] at bc
    /-
      case refine_2
      N : Type u_2
      r : N → N → Prop
      inst✝ : Group N
      h : Contravariant N N (Function.swap fun x1 x2 => HMul.hMul x1 x2) r
      a b c : N
      bc : r (HMul.hMul (HMul.hMul b a) (Inv.inv a)) (HMul.hMul (HMul.hMul c a) (Inv …
      ⊢ r (Function.swap (fun x1 x2 => HMul.hMul x1 x2) a b) (Function.swap (fun x1  …
    -/
    exact h a⁻¹ bc
    /-
      🎉 no goals
    -/



@[to_additive]
instance (priority := 100) Group.covconv_swap [Group N] [CovariantClass N N (swap (· * ·)) r] :
    ContravariantClass N N (swap (· * ·)) r :=
  ⟨Group.covariant_swap_iff_contravariant_swap.mp CovariantClass.elim⟩


@[to_additive]
theorem Group.mulRightReflectLE_of_mulRightMono [Group N] [LE N]
    [MulRightMono N] : MulRightReflectLE N :=
  inferInstance


@[to_additive]
theorem Group.mulRightReflectLT_of_mulRightStrictMono [Group N] [LT N]
    [MulRightStrictMono N] : MulRightReflectLT N :=
  inferInstance



theorem act_rel_of_rel_of_act_rel (ab : r a b) (rl : r (μ m b) c) : r (μ m a) c :=
  _root_.trans (act_rel_act_of_rel m ab) rl


theorem rel_act_of_rel_of_rel_act (ab : r a b) (rr : r c (μ m a)) : r c (μ m b) :=
  _root_.trans rr (act_rel_act_of_rel _ ab)


theorem act_rel_act_of_rel_of_rel (ab : r a b) (cd : r c d) : r (mu a c) (mu b d) :=
  _root_.trans (@act_rel_act_of_rel _ _ (swap mu) r _ c _ _ ab) (act_rel_act_of_rel b cd)


theorem rel_of_act_rel_act (m : M) {a b : N} (ab : r (μ m a) (μ m b)) : r a b :=
  ContravariantClass.elim _ ab


theorem act_rel_of_act_rel_of_rel_act_rel (ab : r (μ m a) b) (rl : r (μ m b) (μ m c)) :
    r (μ m a) c :=
  _root_.trans ab (rel_of_act_rel_act m rl)


theorem rel_act_of_act_rel_act_of_rel_act (ab : r (μ m a) (μ m b)) (rr : r b (μ m c)) :
    r a (μ m c) :=
  _root_.trans (rel_of_act_rel_act m ab) rr


/-- The partial application of a constant to a covariant operator is monotone. -/
theorem Covariant.monotone_of_const [CovariantClass M N μ (· ≤ ·)] (m : M) : Monotone (μ m) :=
  fun _ _ ↦ CovariantClass.elim m


/-- A monotone function remains monotone when composed with the partial application
of a covariant operator. E.g., `∀ (m : ℕ), Monotone f → Monotone (fun n ↦ f (m + n))`. -/
theorem Monotone.covariant_of_const [CovariantClass M N μ (· ≤ ·)] (hf : Monotone f) (m : M) :
    Monotone (f <| μ m ·) :=
  hf.comp (Covariant.monotone_of_const m)


/-- Same as `Monotone.covariant_of_const`, but with the constant on the other side of
the operator.  E.g., `∀ (m : ℕ), Monotone f → Monotone (fun n ↦ f (n + m))`. -/
theorem Monotone.covariant_of_const' {μ : N → N → N} [CovariantClass N N (swap μ) (· ≤ ·)]
    (hf : Monotone f) (m : N) : Monotone (f <| μ · m) :=
  Monotone.covariant_of_const (μ := swap μ) hf m


/-- Dual of `Monotone.covariant_of_const` -/
theorem Antitone.covariant_of_const [CovariantClass M N μ (· ≤ ·)] (hf : Antitone f) (m : M) :
    Antitone (f <| μ m ·) :=
  hf.comp_monotone <| Covariant.monotone_of_const m


/-- Dual of `Monotone.covariant_of_const'` -/
theorem Antitone.covariant_of_const' {μ : N → N → N} [CovariantClass N N (swap μ) (· ≤ ·)]
    (hf : Antitone f) (m : N) : Antitone (f <| μ · m) :=
  Antitone.covariant_of_const (μ := swap μ) hf m


theorem covariant_le_of_covariant_lt [PartialOrder N] :
    Covariant M N μ (· < ·) → Covariant M N μ (· ≤ ·) := by
  /-
    M : Type u_1
    N : Type u_2
    μ : M → N → N
    inst✝ : PartialOrder N
    ⊢ (Covariant M N μ fun x1 x2 => LT.lt x1 x2) → Covariant M N μ fun x1 x2 => LE …
  -/
  intro h a b c bc
  /-
    M : Type u_1
    N : Type u_2
    μ : M → N → N
    inst✝ : PartialOrder N
    h : Covariant M N μ fun x1 x2 => LT.lt x1 x2
    a : M
    b c : N
    bc : LE.le b c
    ⊢ LE.le (μ a b) (μ a c)
  -/
  rcases bc.eq_or_lt with (rfl | bc)
    /-
      case inl
      M : Type u_1
      N : Type u_2
      μ : M → N → N
      inst✝ : PartialOrder N
      h : Covariant M N μ fun x1 x2 => LT.lt x1 x2
      a : M
      b : N
      bc : LE.le b b
      ⊢ LE.le (μ a b) (μ a b)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      M : Type u_1
      N : Type u_2
      μ : M → N → N
      inst✝ : PartialOrder N
      h : Covariant M N μ fun x1 x2 => LT.lt x1 x2
      a : M
      b c : N
      bc✝ : LE.le b c
      bc : LT.lt b c
      ⊢ LE.le (μ a b) (μ a c)
    -/
  · exact (h _ bc).le
    /-
      🎉 no goals
    -/


theorem covariantClass_le_of_lt [PartialOrder N] [CovariantClass M N μ (· < ·)] :
    CovariantClass M N μ (· ≤ ·) := ⟨covariant_le_of_covariant_lt _ _ _ CovariantClass.elim⟩


@[to_additive]
theorem mulLeftMono_of_mulLeftStrictMono (M) [Mul M] [PartialOrder M] [MulLeftStrictMono M] :
    MulLeftMono M := covariantClass_le_of_lt _ _ _


@[to_additive]
theorem mulRightMono_of_mulRightStrictMono (M) [Mul M] [PartialOrder M] [MulRightStrictMono M] :
    MulRightMono M := covariantClass_le_of_lt _ _ _


theorem contravariant_le_iff_contravariant_lt_and_eq [PartialOrder N] :
    Contravariant M N μ (· ≤ ·) ↔ Contravariant M N μ (· < ·) ∧ Contravariant M N μ (· = ·) := by
  /-
    M : Type u_1
    N : Type u_2
    μ : M → N → N
    inst✝ : PartialOrder N
    ⊢ Iff (Contravariant M N μ fun x1 x2 => LE.le x1 x2) (And (Contravariant M N μ …
  -/
  refine ⟨fun h ↦ ⟨fun a b c bc ↦ ?_, fun a b c bc ↦ ?_⟩, fun h ↦ fun a b c bc ↦ ?_⟩
    /-
      case refine_1
      M : Type u_1
      N : Type u_2
      μ : M → N → N
      inst✝ : PartialOrder N
      h : Contravariant M N μ fun x1 x2 => LE.le x1 x2
      a : M
      b c : N
      bc : (fun x1 x2 => LT.lt x1 x2) (μ a b) (μ a c)
      ⊢ (fun x1 x2 => LT.lt x1 x2) b c
    -/
  · exact (h a bc.le).lt_of_ne (by rintro rfl; exact lt_irrefl _ bc)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      N : Type u_2
      μ : M → N → N
      inst✝ : PartialOrder N
      h : Contravariant M N μ fun x1 x2 => LE.le x1 x2
      a : M
      b c : N
      bc : (fun x1 x2 => Eq x1 x2) (μ a b) (μ a c)
      ⊢ (fun x1 x2 => Eq x1 x2) b c
    -/
  · exact (h a bc.le).antisymm (h a bc.ge)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      M : Type u_1
      N : Type u_2
      μ : M → N → N
      inst✝ : PartialOrder N
      h : And (Contravariant M N μ fun x1 x2 => LT.lt x1 x2) (Contravariant M N μ fu …
      a : M
      b c : N
      bc : (fun x1 x2 => LE.le x1 x2) (μ a b) (μ a c)
      ⊢ (fun x1 x2 => LE.le x1 x2) b c
    -/
  · exact bc.lt_or_eq.elim (fun bc ↦ (h.1 a bc).le) (fun bc ↦ (h.2 a bc).le)
    /-
      🎉 no goals
    -/


theorem contravariant_lt_of_contravariant_le [PartialOrder N] :
    Contravariant M N μ (· ≤ ·) → Contravariant M N μ (· < ·) :=
  And.left ∘ (contravariant_le_iff_contravariant_lt_and_eq M N μ).mp


theorem covariant_le_iff_contravariant_lt [LinearOrder N] :
    Covariant M N μ (· ≤ ·) ↔ Contravariant M N μ (· < ·) :=
  ⟨fun h _ _ _ bc ↦ not_le.mp fun k ↦ bc.not_le (h _ k),
   fun h _ _ _ bc ↦ not_lt.mp fun k ↦ bc.not_lt (h _ k)⟩


theorem covariant_lt_iff_contravariant_le [LinearOrder N] :
    Covariant M N μ (· < ·) ↔ Contravariant M N μ (· ≤ ·) :=
  ⟨fun h _ _ _ bc ↦ not_lt.mp fun k ↦ bc.not_lt (h _ k),
   fun h _ _ _ bc ↦ not_le.mp fun k ↦ bc.not_le (h _ k)⟩


theorem covariant_flip_iff [h : Std.Commutative mu] :
                                                         /-
                                                           N : Type u_2
                                                           r : N → N → Prop
                                                           mu : N → N → N
                                                           h : Std.Commutative mu
                                                           ⊢ Iff (Covariant N N (flip mu) r) (Covariant N N mu r)
                                                         -/
    Covariant N N (flip mu) r ↔ Covariant N N mu r := by unfold flip; simp_rw [h.comm]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem contravariant_flip_iff [h : Std.Commutative mu] :
                                                                 /-
                                                                   N : Type u_2
                                                                   r : N → N → Prop
                                                                   mu : N → N → N
                                                                   h : Std.Commutative mu
                                                                   ⊢ Iff (Contravariant N N (flip mu) r) (Contravariant N N mu r)
                                                                 -/
    Contravariant N N (flip mu) r ↔ Contravariant N N mu r := by unfold flip; simp_rw [h.comm]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance contravariant_lt_of_covariant_le [LinearOrder N]
    [CovariantClass N N mu (· ≤ ·)] : ContravariantClass N N mu (· < ·) where
  elim := (covariant_le_iff_contravariant_lt N N mu).mp CovariantClass.elim


@[to_additive]
theorem mulLeftReflectLT_of_mulLeftMono [Mul N] [LinearOrder N] [MulLeftMono N] :
    MulLeftReflectLT N :=
  inferInstance


@[to_additive]
theorem mulRightReflectLT_of_mulRightMono [Mul N] [LinearOrder N] [MulRightMono N] :
    MulRightReflectLT N :=
  inferInstance


instance covariant_lt_of_contravariant_le [LinearOrder N]
    [ContravariantClass N N mu (· ≤ ·)] : CovariantClass N N mu (· < ·) where
  elim := (covariant_lt_iff_contravariant_le N N mu).mpr ContravariantClass.elim


@[to_additive]
theorem mulLeftStrictMono_of_mulLeftReflectLE [Mul N] [LinearOrder N] [MulLeftReflectLE N] :
    MulLeftStrictMono N :=
  inferInstance


@[to_additive]
theorem mulRightStrictMono_of_mulRightReflectLE [Mul N] [LinearOrder N] [MulRightReflectLE N] :
    MulRightStrictMono N :=
  inferInstance


@[to_additive]
instance covariant_swap_mul_of_covariant_mul [CommSemigroup N]
    [CovariantClass N N (· * ·) r] : CovariantClass N N (swap (· * ·)) r where
  elim := (covariant_flip_iff N r (· * ·)).mpr CovariantClass.elim


@[to_additive]
theorem mulRightMono_of_mulLeftMono [CommSemigroup N] [LE N] [MulLeftMono N] :
    MulRightMono N :=
  inferInstance


@[to_additive]
theorem mulRightStrictMono_of_mulLeftStrictMono [CommSemigroup N] [LT N] [MulLeftStrictMono N] :
    MulRightStrictMono N :=
  inferInstance


@[to_additive]
instance contravariant_swap_mul_of_contravariant_mul [CommSemigroup N]
    [ContravariantClass N N (· * ·) r] : ContravariantClass N N (swap (· * ·)) r where
  elim := (contravariant_flip_iff N r (· * ·)).mpr ContravariantClass.elim


@[to_additive]
theorem mulRightReflectLE_of_mulLeftReflectLE [CommSemigroup N] [LE N] [MulLeftReflectLE N] :
    MulRightReflectLE N :=
  inferInstance


@[to_additive]
theorem mulRightReflectLT_of_mulLeftReflectLT [CommSemigroup N] [LT N] [MulLeftReflectLT N] :
    MulRightReflectLT N :=
  inferInstance


theorem covariant_lt_of_covariant_le_of_contravariant_eq [ContravariantClass M N μ (· = ·)]
    [PartialOrder N] [CovariantClass M N μ (· ≤ ·)] : CovariantClass M N μ (· < ·) where
  elim a _ _ bc := (CovariantClass.elim a bc.le).lt_of_ne (bc.ne ∘ ContravariantClass.elim _)


theorem contravariant_le_of_contravariant_eq_and_lt [PartialOrder N]
    [ContravariantClass M N μ (· = ·)] [ContravariantClass M N μ (· < ·)] :
    ContravariantClass M N μ (· ≤ ·) where
  elim := (contravariant_le_iff_contravariant_lt_and_eq M N μ).mpr
    ⟨ContravariantClass.elim, ContravariantClass.elim⟩

/- TODO:
  redefine `IsLeftCancel N mu` as abbrev of `ContravariantClass N N mu (· = ·)`,
  redefine `IsRightCancel N mu` as abbrev of `ContravariantClass N N (flip mu) (· = ·)`,
  redefine `IsLeftCancelMul` as abbrev of `IsLeftCancel`,
  then the following four instances (actually eight) can be removed in favor of the above two. -/


@[to_additive]
instance IsLeftCancelMul.mulLeftStrictMono_of_mulLeftMono [Mul N] [IsLeftCancelMul N]
    [PartialOrder N] [MulLeftMono N] :
    MulLeftStrictMono N where
  elim a _ _ bc := (CovariantClass.elim a bc.le).lt_of_ne ((mul_ne_mul_right a).mpr bc.ne)


@[to_additive]
instance IsRightCancelMul.mulRightStrictMono_of_mulRightMono
    [Mul N] [IsRightCancelMul N] [PartialOrder N] [MulRightMono N] :
    MulRightStrictMono N where
  elim a _ _ bc := (CovariantClass.elim a bc.le).lt_of_ne ((mul_ne_mul_left a).mpr bc.ne)


@[to_additive]
instance IsLeftCancelMul.mulLeftReflectLE_of_mulLeftReflectLT [Mul N] [IsLeftCancelMul N]
    [PartialOrder N] [MulLeftReflectLT N] :
    MulLeftReflectLE N where
  elim := (contravariant_le_iff_contravariant_lt_and_eq N N _).mpr
    ⟨ContravariantClass.elim, fun _ ↦ mul_left_cancel⟩


@[to_additive]
instance IsRightCancelMul.mulRightReflectLE_of_mulRightReflectLT
    [Mul N] [IsRightCancelMul N] [PartialOrder N] [MulRightReflectLT N] :
    MulRightReflectLE N where
  elim := (contravariant_le_iff_contravariant_lt_and_eq N N _).mpr
    ⟨ContravariantClass.elim, fun _ ↦ mul_right_cancel⟩


