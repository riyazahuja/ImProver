/-- A function `f` from a real normed space is uniformly convex with modulus `φ` if
`f (t • x + (1 - t) • y) ≤ t • f x + (1 - t) • f y - t * (1 - t) * φ ‖x - y‖` for all `t ∈ [0, 1]`.

`φ` is usually taken to be a monotone function such that `φ r = 0 ↔ r = 0`. -/
def UniformConvexOn (s : Set E) (φ : ℝ → ℝ) (f : E → ℝ) : Prop :=
  Convex ℝ s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : ℝ⦄, 0 ≤ a → 0 ≤ b → a + b = 1 →
    f (a • x + b • y) ≤ a • f x + b • f y - a * b * φ ‖x - y‖


/-- A function `f` from a real normed space is uniformly concave with modulus `φ` if
`t • f x + (1 - t) • f y + t * (1 - t) * φ ‖x - y‖ ≤ f (t • x + (1 - t) • y)` for all `t ∈ [0, 1]`.

`φ` is usually taken to be a monotone function such that `φ r = 0 ↔ r = 0`. -/
def UniformConcaveOn (s : Set E) (φ : ℝ → ℝ) (f : E → ℝ) : Prop :=
  Convex ℝ s ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → ∀ ⦃a b : ℝ⦄, 0 ≤ a → 0 ≤ b → a + b = 1 →
    a • f x + b • f y + a * b * φ ‖x - y‖ ≤ f (a • x + b • y)


@[simp] lemma uniformConvexOn_zero : UniformConvexOn s 0 f ↔ ConvexOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    f : E → Real
    ⊢ Iff (UniformConvexOn s 0 f) (ConvexOn Real s f)
  -/
  simp [UniformConvexOn, ConvexOn]
  /-
    🎉 no goals
  -/


@[simp] lemma uniformConcaveOn_zero : UniformConcaveOn s 0 f ↔ ConcaveOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    f : E → Real
    ⊢ Iff (UniformConcaveOn s 0 f) (ConcaveOn Real s f)
  -/
  simp [UniformConcaveOn, ConcaveOn]
  /-
    🎉 no goals
  -/


protected alias ⟨_, ConvexOn.uniformConvexOn_zero⟩ := uniformConvexOn_zero

protected alias ⟨_, ConcaveOn.uniformConcaveOn_zero⟩ := uniformConcaveOn_zero


lemma UniformConvexOn.mono (hψφ : ψ ≤ φ) (hf : UniformConvexOn s φ f) : UniformConvexOn s ψ f :=
                                                                          /-
                                                                            E : Type u_1
                                                                            inst✝¹ : NormedAddCommGroup E
                                                                            inst✝ : NormedSpace Real E
                                                                            φ ψ : Real → Real
                                                                            s : Set E
                                                                            f : E → Real
                                                                            hψφ : LE.le ψ φ
                                                                            hf : UniformConvexOn s φ f
                                                                            x : E
                                                                            hx : Membership.mem s x
                                                                            y : E
                                                                            hy : Membership.mem s y
                                                                            a b : Real
                                                                            ha : LE.le 0 a
                                                                            hb : LE.le 0 b
                                                                            hab : Eq (HAdd.hAdd a b) 1
                                                                            ⊢ LE.le (HSub.hSub (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HM …
                                                                          -/
  ⟨hf.1, fun x hx y hy a b ha hb hab ↦ (hf.2 hx hy ha hb hab).trans <| by gcongr; apply hψφ⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma UniformConcaveOn.mono (hψφ : ψ ≤ φ) (hf : UniformConcaveOn s φ f) : UniformConcaveOn s ψ f :=
                                                                           /-
                                                                             E : Type u_1
                                                                             inst✝¹ : NormedAddCommGroup E
                                                                             inst✝ : NormedSpace Real E
                                                                             φ ψ : Real → Real
                                                                             s : Set E
                                                                             f : E → Real
                                                                             hψφ : LE.le ψ φ
                                                                             hf : UniformConcaveOn s φ f
                                                                             x : E
                                                                             hx : Membership.mem s x
                                                                             y : E
                                                                             hy : Membership.mem s y
                                                                             a b : Real
                                                                             ha : LE.le 0 a
                                                                             hb : LE.le 0 b
                                                                             hab : Eq (HAdd.hAdd a b) 1
                                                                             ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HM …
                                                                           -/
  ⟨hf.1, fun x hx y hy a b ha hb hab ↦ (hf.2 hx hy ha hb hab).trans' <| by gcongr; apply hψφ⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma UniformConvexOn.convexOn (hf : UniformConvexOn s φ f) (hφ : 0 ≤ φ) : ConvexOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    hφ : LE.le 0 φ
    ⊢ ConvexOn Real s f
  -/
  simpa using hf.mono hφ
  /-
    🎉 no goals
  -/


lemma UniformConcaveOn.concaveOn (hf : UniformConcaveOn s φ f) (hφ : 0 ≤ φ) : ConcaveOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConcaveOn s φ f
    hφ : LE.le 0 φ
    ⊢ ConcaveOn Real s f
  -/
  simpa using hf.mono hφ
  /-
    🎉 no goals
  -/


lemma UniformConvexOn.strictConvexOn (hf : UniformConvexOn s φ f) (hφ : ∀ r, r ≠ 0 → 0 < φ r) :
    StrictConvexOn ℝ s f := by
  refine ⟨hf.1, fun x hx y hy hxy a b ha hb hab ↦ (hf.2 hx hy ha.le hb.le hab).trans_lt <|
    sub_lt_self _ ?_⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  rw [← sub_ne_zero, ← norm_pos_iff] at hxy
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : LT.lt 0 (Norm.norm (HSub.hSub x y))
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  have := hφ _ hxy.ne'
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : LT.lt 0 (Norm.norm (HSub.hSub x y))
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    this : LT.lt 0 (φ (Norm.norm (HSub.hSub x y)))
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma UniformConcaveOn.strictConcaveOn (hf : UniformConcaveOn s φ f) (hφ : ∀ r, r ≠ 0 → 0 < φ r) :
    StrictConcaveOn ℝ s f := by
  refine ⟨hf.1, fun x hx y hy hxy a b ha hb hab ↦ (hf.2 hx hy ha.le hb.le hab).trans_lt' <|
    lt_add_of_pos_right _ ?_⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConcaveOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  rw [← sub_ne_zero, ← norm_pos_iff] at hxy
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConcaveOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : LT.lt 0 (Norm.norm (HSub.hSub x y))
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  have := hφ _ hxy.ne'
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConcaveOn s φ f
    hφ : ∀ (r : Real), Ne r 0 → LT.lt 0 (φ r)
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hxy : LT.lt 0 (Norm.norm (HSub.hSub x y))
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    this : LT.lt 0 (φ (Norm.norm (HSub.hSub x y)))
    ⊢ LT.lt 0 (HMul.hMul (HMul.hMul a b) (φ (Norm.norm (HSub.hSub x y))))
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma UniformConvexOn.add (hf : UniformConvexOn s φ f) (hg : UniformConvexOn s ψ g) :
    UniformConvexOn s (φ + ψ) (f + g) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ ψ : Real → Real
    s : Set E
    f g : E → Real
    hf : UniformConvexOn s φ f
    hg : UniformConvexOn s ψ g
    ⊢ UniformConvexOn s (HAdd.hAdd φ ψ) (HAdd.hAdd f g)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ ?_⟩
  simpa [mul_add, add_add_add_comm, sub_add_sub_comm]
    using add_le_add (hf.2 hx hy ha hb hab) (hg.2 hx hy ha hb hab)


lemma UniformConcaveOn.add (hf : UniformConcaveOn s φ f) (hg : UniformConcaveOn s ψ g) :
    UniformConcaveOn s (φ + ψ) (f + g) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ ψ : Real → Real
    s : Set E
    f g : E → Real
    hf : UniformConcaveOn s φ f
    hg : UniformConcaveOn s ψ g
    ⊢ UniformConcaveOn s (HAdd.hAdd φ ψ) (HAdd.hAdd f g)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ ?_⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ ψ : Real → Real
    s : Set E
    f g : E → Real
    hf : UniformConcaveOn s φ f
    hg : UniformConcaveOn s ψ g
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (HAdd.hAdd f g x)) (HSMul.hSMul b …
  -/
  simpa [mul_add, add_add_add_comm] using add_le_add (hf.2 hx hy ha hb hab) (hg.2 hx hy ha hb hab)
  /-
    🎉 no goals
  -/


lemma UniformConvexOn.neg (hf : UniformConvexOn s φ f) : UniformConcaveOn s φ (-f) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    ⊢ UniformConcaveOn s φ (Neg.neg f)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ le_of_neg_le_neg ?_⟩
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConvexOn s φ f
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (Neg.neg (Neg.neg f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))))  …
  -/
  simpa [add_comm, -neg_le_neg_iff, le_sub_iff_add_le'] using hf.2 hx hy ha hb hab
  /-
    🎉 no goals
  -/


lemma UniformConcaveOn.neg (hf : UniformConcaveOn s φ f) : UniformConvexOn s φ (-f) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    φ : Real → Real
    s : Set E
    f : E → Real
    hf : UniformConcaveOn s φ f
    ⊢ UniformConvexOn s φ (Neg.neg f)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ le_of_neg_le_neg ?_⟩
  simpa [add_comm, -neg_le_neg_iff, ← le_sub_iff_add_le', sub_eq_add_neg, neg_add]
    using hf.2 hx hy ha hb hab


lemma UniformConvexOn.sub (hf : UniformConvexOn s φ f) (hg : UniformConcaveOn s ψ g) :
                                            /-
                                              E : Type u_1
                                              inst✝¹ : NormedAddCommGroup E
                                              inst✝ : NormedSpace Real E
                                              φ ψ : Real → Real
                                              s : Set E
                                              f g : E → Real
                                              hf : UniformConvexOn s φ f
                                              hg : UniformConcaveOn s ψ g
                                              ⊢ UniformConvexOn s (HAdd.hAdd φ ψ) (HSub.hSub f g)
                                            -/
    UniformConvexOn s (φ + ψ) (f - g) := by simpa using hf.add hg.neg
                                            /-
                                              🎉 no goals
                                            -/


lemma UniformConcaveOn.sub (hf : UniformConcaveOn s φ f) (hg : UniformConvexOn s ψ g) :
                                             /-
                                               E : Type u_1
                                               inst✝¹ : NormedAddCommGroup E
                                               inst✝ : NormedSpace Real E
                                               φ ψ : Real → Real
                                               s : Set E
                                               f g : E → Real
                                               hf : UniformConcaveOn s φ f
                                               hg : UniformConvexOn s ψ g
                                               ⊢ UniformConcaveOn s (HAdd.hAdd φ ψ) (HSub.hSub f g)
                                             -/
    UniformConcaveOn s (φ + ψ) (f - g) := by simpa using hf.add hg.neg
                                             /-
                                               🎉 no goals
                                             -/


/-- A function `f` from a real normed space is `m`-strongly convex if it is uniformly convex with
modulus `φ(r) = m / 2 * r ^ 2`.

In an inner product space, this is equivalent to `x ↦ f x - m / 2 * ‖x‖ ^ 2` being convex. -/
def StrongConvexOn (s : Set E) (m : ℝ) : (E → ℝ) → Prop :=
  UniformConvexOn s fun r ↦ m / (2 : ℝ) * r ^ 2


/-- A function `f` from a real normed space is `m`-strongly concave if is strongly concave with
modulus `φ(r) = m / 2 * r ^ 2`.

In an inner product space, this is equivalent to `x ↦ f x + m / 2 * ‖x‖ ^ 2` being concave. -/
def StrongConcaveOn (s : Set E) (m : ℝ) : (E → ℝ) → Prop :=
  UniformConcaveOn s fun r ↦ m / (2 : ℝ) * r ^ 2


nonrec lemma StrongConvexOn.mono (hmn : m ≤ n) (hf : StrongConvexOn s n f) : StrongConvexOn s m f :=
                     /-
                       E : Type u_1
                       inst✝¹ : NormedAddCommGroup E
                       inst✝ : NormedSpace Real E
                       s : Set E
                       f : E → Real
                       m n : Real
                       hmn : LE.le m n
                       hf : StrongConvexOn s n f
                       r : Real
                       ⊢ LE.le (HMul.hMul (HDiv.hDiv m 2) (HPow.hPow r 2)) (HMul.hMul (HDiv.hDiv n 2) …
                     -/
  hf.mono fun r ↦ by gcongr
                     /-
                       🎉 no goals
                     -/


nonrec lemma StrongConcaveOn.mono (hmn : m ≤ n) (hf : StrongConcaveOn s n f) :
                                                /-
                                                  E : Type u_1
                                                  inst✝¹ : NormedAddCommGroup E
                                                  inst✝ : NormedSpace Real E
                                                  s : Set E
                                                  f : E → Real
                                                  m n : Real
                                                  hmn : LE.le m n
                                                  hf : StrongConcaveOn s n f
                                                  r : Real
                                                  ⊢ LE.le (HMul.hMul (HDiv.hDiv m 2) (HPow.hPow r 2)) (HMul.hMul (HDiv.hDiv n 2) …
                                                -/
    StrongConcaveOn s m f := hf.mono fun r ↦ by gcongr
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma strongConvexOn_zero : StrongConvexOn s 0 f ↔ ConvexOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    f : E → Real
    ⊢ Iff (StrongConvexOn s 0 f) (ConvexOn Real s f)
  -/
  simp [StrongConvexOn, ← Pi.zero_def]
  /-
    🎉 no goals
  -/


@[simp] lemma strongConcaveOn_zero : StrongConcaveOn s 0 f ↔ ConcaveOn ℝ s f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    f : E → Real
    ⊢ Iff (StrongConcaveOn s 0 f) (ConcaveOn Real s f)
  -/
  simp [StrongConcaveOn, ← Pi.zero_def]
  /-
    🎉 no goals
  -/


nonrec lemma StrongConvexOn.strictConvexOn (hf : StrongConvexOn s m f) (hm : 0 < m) :
                                                            /-
                                                              E : Type u_1
                                                              inst✝¹ : NormedAddCommGroup E
                                                              inst✝ : NormedSpace Real E
                                                              s : Set E
                                                              f : E → Real
                                                              m : Real
                                                              hf : StrongConvexOn s m f
                                                              hm : LT.lt 0 m
                                                              r : Real
                                                              hr : Ne r 0
                                                              ⊢ LT.lt 0 (HMul.hMul (HDiv.hDiv m 2) (HPow.hPow r 2))
                                                            -/
    StrictConvexOn ℝ s f := hf.strictConvexOn fun r hr ↦ by positivity
                                                            /-
                                                              🎉 no goals
                                                            -/


nonrec lemma StrongConcaveOn.strictConcaveOn (hf : StrongConcaveOn s m f) (hm : 0 < m) :
                                                              /-
                                                                E : Type u_1
                                                                inst✝¹ : NormedAddCommGroup E
                                                                inst✝ : NormedSpace Real E
                                                                s : Set E
                                                                f : E → Real
                                                                m : Real
                                                                hf : StrongConcaveOn s m f
                                                                hm : LT.lt 0 m
                                                                r : Real
                                                                hr : Ne r 0
                                                                ⊢ LT.lt 0 (HMul.hMul (HDiv.hDiv m 2) (HPow.hPow r 2))
                                                              -/
    StrictConcaveOn ℝ s f := hf.strictConcaveOn fun r hr ↦ by positivity
                                                              /-
                                                                🎉 no goals
                                                              -/


private lemma aux_sub (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a * (f x - m / (2 : ℝ) * ‖x‖ ^ 2) + b * (f y - m / (2 : ℝ) * ‖y‖ ^ 2) +
      m / (2 : ℝ) * ‖a • x + b • y‖ ^ 2
      = a * f x + b * f y - m / (2 : ℝ) * a * b * ‖x - y‖ ^ 2 := by
  rw [norm_add_sq_real, norm_sub_sq_real, norm_smul, norm_smul, real_inner_smul_left,
    inner_smul_right, norm_of_nonneg ha, norm_of_nonneg hb, mul_pow, mul_pow]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    a b m : Real
    x y : E
    f : E → Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul a (HSub.hSub (f x) (HMul.hMul (HDiv.hDiv …
  -/
  obtain rfl := eq_sub_of_add_eq hab
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    b m : Real
    x y : E
    f : E → Real
    hb : LE.le 0 b
    ha : LE.le 0 (HSub.hSub 1 b)
    hab : Eq (HAdd.hAdd (HSub.hSub 1 b) b) 1
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HSub.hSub 1 b) (HSub.hSub (f x) (HMul.h …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


private lemma aux_add (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a * (f x + m / (2 : ℝ) * ‖x‖ ^ 2) + b * (f y + m / (2 : ℝ) * ‖y‖ ^ 2) -
      m / (2 : ℝ) * ‖a • x + b • y‖ ^ 2
      = a * f x + b * f y + m / (2 : ℝ) * a * b * ‖x - y‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    a b m : Real
    x y : E
    f : E → Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul a (HAdd.hAdd (f x) (HMul.hMul (HDiv.hDiv …
  -/
  simpa [neg_div] using aux_sub (E := E) (m := -m) ha hb hab
  /-
    🎉 no goals
  -/


lemma strongConvexOn_iff_convex :
    StrongConvexOn s m f ↔ ConvexOn ℝ s fun x ↦ f x - m / (2 : ℝ) * ‖x‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    s : Set E
    m : Real
    f : E → Real
    ⊢ Iff (StrongConvexOn s m f) (ConvexOn Real s fun x => HSub.hSub (f x) (HMul.h …
  -/
  refine and_congr_right fun _ ↦ forall₄_congr fun x _ y _ ↦ forall₅_congr fun a b ha hb hab ↦ ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    s : Set E
    m : Real
    f : E → Real
    x✝² : Convex Real s
    x : E
    x✝¹ : Membership.mem s x
    y : E
    x✝ : Membership.mem s y
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Iff (LE.le (f (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HSub.hSub (H …
  -/
  simp_rw [sub_le_iff_le_add, smul_eq_mul, aux_sub ha hb hab, mul_assoc, mul_left_comm]
  /-
    🎉 no goals
  -/


lemma strongConcaveOn_iff_convex :
    StrongConcaveOn s m f ↔ ConcaveOn ℝ s fun x ↦ f x + m / (2 : ℝ) * ‖x‖ ^ 2 := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    s : Set E
    m : Real
    f : E → Real
    ⊢ Iff (StrongConcaveOn s m f) (ConcaveOn Real s fun x => HAdd.hAdd (f x) (HMul …
  -/
  refine and_congr_right fun _ ↦ forall₄_congr fun x _ y _ ↦ forall₅_congr fun a b ha hb hab ↦ ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    s : Set E
    m : Real
    f : E → Real
    x✝² : Convex Real s
    x : E
    x✝¹ : Membership.mem s x
    y : E
    x✝ : Membership.mem s y
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Iff (LE.le (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y)) …
  -/
  simp_rw [← sub_le_iff_le_add, smul_eq_mul, aux_add ha hb hab, mul_assoc, mul_left_comm]
  /-
    🎉 no goals
  -/


