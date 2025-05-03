/-- We say that a type has `MeasurableAdd` if `(· + c)` and `(· + c)` are measurable functions.
For a typeclass assuming measurability of `uncurry (· + ·)` see `MeasurableAdd₂`. -/
class MeasurableAdd (M : Type*) [MeasurableSpace M] [Add M] : Prop where
  measurable_const_add : ∀ c : M, Measurable (c + ·)
  measurable_add_const : ∀ c : M, Measurable (· + c)


/-- We say that a type has `MeasurableAdd₂` if `uncurry (· + ·)` is a measurable functions.
For a typeclass assuming measurability of `(c + ·)` and `(· + c)` see `MeasurableAdd`. -/
class MeasurableAdd₂ (M : Type*) [MeasurableSpace M] [Add M] : Prop where
  measurable_add : Measurable fun p : M × M => p.1 + p.2


/-- We say that a type has `MeasurableMul` if `(c * ·)` and `(· * c)` are measurable functions.
For a typeclass assuming measurability of `uncurry (*)` see `MeasurableMul₂`. -/
@[to_additive]
class MeasurableMul (M : Type*) [MeasurableSpace M] [Mul M] : Prop where
  measurable_const_mul : ∀ c : M, Measurable (c * ·)
  measurable_mul_const : ∀ c : M, Measurable (· * c)


/-- We say that a type has `MeasurableMul₂` if `uncurry (· * ·)` is a measurable functions.
For a typeclass assuming measurability of `(c * ·)` and `(· * c)` see `MeasurableMul`. -/
@[to_additive MeasurableAdd₂]
class MeasurableMul₂ (M : Type*) [MeasurableSpace M] [Mul M] : Prop where
  measurable_mul : Measurable fun p : M × M => p.1 * p.2


@[to_additive (attr := fun_prop, measurability)]
theorem Measurable.const_mul [MeasurableMul M] (hf : Measurable f) (c : M) :
    Measurable fun x => c * f x :=
  (measurable_const_mul c).comp hf


@[to_additive (attr := measurability)]
theorem AEMeasurable.const_mul [MeasurableMul M] (hf : AEMeasurable f μ) (c : M) :
    AEMeasurable (fun x => c * f x) μ :=
  (MeasurableMul.measurable_const_mul c).comp_aemeasurable hf


@[to_additive (attr := measurability)]
theorem Measurable.mul_const [MeasurableMul M] (hf : Measurable f) (c : M) :
    Measurable fun x => f x * c :=
  (measurable_mul_const c).comp hf


@[to_additive (attr := measurability)]
theorem AEMeasurable.mul_const [MeasurableMul M] (hf : AEMeasurable f μ) (c : M) :
    AEMeasurable (fun x => f x * c) μ :=
  (measurable_mul_const c).comp_aemeasurable hf


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem Measurable.mul [MeasurableMul₂ M] (hf : Measurable f) (hg : Measurable g) :
    Measurable fun a => f a * g a :=
  measurable_mul.comp (hf.prod_mk hg)


/-- Compositional version of `Measurable.mul` for use by `fun_prop`. -/
@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))
"Compositional version of `Measurable.add` for use by `fun_prop`."]
lemma Measurable.mul' [MeasurableMul₂ M] {f g : α → β → M} {h : α → β} (hf : Measurable ↿f)
    (hg : Measurable ↿g) (hh : Measurable h) : Measurable fun a ↦ (f a * g a) (h a) := by
  /-
    M : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝² : MeasurableSpace M
    inst✝¹ : Mul M
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableMul₂ M
    f g : α → β → M
    h : α → β
    hf : Measurable (Function.HasUncurry.uncurry f)
    hg : Measurable (Function.HasUncurry.uncurry g)
    hh : Measurable h
    ⊢ Measurable fun a => HMul.hMul (f a) (g a) (h a)
  -/
  simp; fun_prop
        /-
          🎉 no goals
        -/


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem AEMeasurable.mul' [MeasurableMul₂ M] (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (f * g) μ :=
  measurable_mul.comp_aemeasurable (hf.prod_mk hg)


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem AEMeasurable.mul [MeasurableMul₂ M] (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (fun a => f a * g a) μ :=
  measurable_mul.comp_aemeasurable (hf.prod_mk hg)


@[to_additive]
instance (priority := 100) MeasurableMul₂.toMeasurableMul [MeasurableMul₂ M] :
    MeasurableMul M :=
  ⟨fun _ => measurable_const.mul measurable_id, fun _ => measurable_id.mul measurable_const⟩


@[to_additive]
instance Pi.measurableMul {ι : Type*} {α : ι → Type*} [∀ i, Mul (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableMul (α i)] : MeasurableMul (∀ i, α i) :=
  ⟨fun _ => measurable_pi_iff.mpr fun i => (measurable_pi_apply i).const_mul _, fun _ =>
    measurable_pi_iff.mpr fun i => (measurable_pi_apply i).mul_const _⟩


@[to_additive Pi.measurableAdd₂]
instance Pi.measurableMul₂ {ι : Type*} {α : ι → Type*} [∀ i, Mul (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableMul₂ (α i)] : MeasurableMul₂ (∀ i, α i) :=
  ⟨measurable_pi_iff.mpr fun _ => measurable_fst.eval.mul measurable_snd.eval⟩


/-- A version of `measurable_div_const` that assumes `MeasurableMul` instead of
  `MeasurableDiv`. This can be nice to avoid unnecessary type-class assumptions. -/
@[to_additive " A version of `measurable_sub_const` that assumes `MeasurableAdd` instead of
  `MeasurableSub`. This can be nice to avoid unnecessary type-class assumptions. "]
theorem measurable_div_const' {G : Type*} [DivInvMonoid G] [MeasurableSpace G] [MeasurableMul G]
                                              /-
                                                G : Type u_2
                                                inst✝² : DivInvMonoid G
                                                inst✝¹ : MeasurableSpace G
                                                inst✝ : MeasurableMul G
                                                g : G
                                                ⊢ Measurable fun h => HDiv.hDiv h g
                                              -/
    (g : G) : Measurable fun h => h / g := by simp_rw [div_eq_mul_inv, measurable_mul_const]
                                              /-
                                                🎉 no goals
                                              -/


/-- This class assumes that the map `β × γ → β` given by `(x, y) ↦ x ^ y` is measurable. -/
class MeasurablePow (β γ : Type*) [MeasurableSpace β] [MeasurableSpace γ] [Pow β γ] : Prop where
  measurable_pow : Measurable fun p : β × γ => p.1 ^ p.2


/-- `Monoid.Pow` is measurable. -/
instance Monoid.measurablePow (M : Type*) [Monoid M] [MeasurableSpace M] [MeasurableMul₂ M] :
    MeasurablePow M ℕ :=
  ⟨measurable_from_prod_countable fun n => by
      /-
        α : Type u_1
        M : Type u_2
        inst✝² : Monoid M
        inst✝¹ : MeasurableSpace M
        inst✝ : MeasurableMul₂ M
        n : Nat
        ⊢ Measurable fun x => HPow.hPow { fst := x, snd := n }.1 { fst := x, snd := n  …
      -/
      induction' n with n ih
        /-
          case zero
          α : Type u_1
          M : Type u_2
          inst✝² : Monoid M
          inst✝¹ : MeasurableSpace M
          inst✝ : MeasurableMul₂ M
          ⊢ Measurable fun x => HPow.hPow { fst := x, snd := 0 }.1 { fst := x, snd := 0  …
        -/
      · simp only [pow_zero, ← Pi.one_def, measurable_one]
        /-
          🎉 no goals
        -/
        /-
          case succ
          α : Type u_1
          M : Type u_2
          inst✝² : Monoid M
          inst✝¹ : MeasurableSpace M
          inst✝ : MeasurableMul₂ M
          n : Nat
          ih : Measurable fun x => HPow.hPow { fst := x, snd := n }.1 { fst := x, snd := …
          ⊢ Measurable fun x => HPow.hPow { fst := x, snd := HAdd.hAdd n 1 }.1 { fst :=  …
        -/
      · simp only [pow_succ]
        /-
          case succ
          α : Type u_1
          M : Type u_2
          inst✝² : Monoid M
          inst✝¹ : MeasurableSpace M
          inst✝ : MeasurableMul₂ M
          n : Nat
          ih : Measurable fun x => HPow.hPow { fst := x, snd := n }.1 { fst := x, snd := …
          ⊢ Measurable fun x => HMul.hMul (HPow.hPow x n) x
        -/
        exact ih.mul measurable_id⟩
        /-
          🎉 no goals
        -/


@[aesop safe 20 apply (rule_sets := [Measurable])]
theorem Measurable.pow (hf : Measurable f) (hg : Measurable g) : Measurable fun x => f x ^ g x :=
  measurable_pow.comp (hf.prod_mk hg)


@[aesop safe 20 apply (rule_sets := [Measurable])]
theorem AEMeasurable.pow (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (fun x => f x ^ g x) μ :=
  measurable_pow.comp_aemeasurable (hf.prod_mk hg)


@[fun_prop, measurability]
theorem Measurable.pow_const (hf : Measurable f) (c : γ) : Measurable fun x => f x ^ c :=
  hf.pow measurable_const


@[fun_prop, measurability]
theorem AEMeasurable.pow_const (hf : AEMeasurable f μ) (c : γ) :
    AEMeasurable (fun x => f x ^ c) μ :=
  hf.pow aemeasurable_const


@[measurability]
theorem Measurable.const_pow (hg : Measurable g) (c : β) : Measurable fun x => c ^ g x :=
  measurable_const.pow hg


@[measurability]
theorem AEMeasurable.const_pow (hg : AEMeasurable g μ) (c : β) :
    AEMeasurable (fun x => c ^ g x) μ :=
  aemeasurable_const.pow hg


/-- We say that a type has `MeasurableSub` if `(c - ·)` and `(· - c)` are measurable
functions. For a typeclass assuming measurability of `uncurry (-)` see `MeasurableSub₂`. -/
class MeasurableSub (G : Type*) [MeasurableSpace G] [Sub G] : Prop where
  measurable_const_sub : ∀ c : G, Measurable (c - ·)
  measurable_sub_const : ∀ c : G, Measurable (· - c)


/-- We say that a type has `MeasurableSub₂` if `uncurry (· - ·)` is a measurable functions.
For a typeclass assuming measurability of `(c - ·)` and `(· - c)` see `MeasurableSub`. -/
class MeasurableSub₂ (G : Type*) [MeasurableSpace G] [Sub G] : Prop where
  measurable_sub : Measurable fun p : G × G => p.1 - p.2


/-- We say that a type has `MeasurableDiv` if `(c / ·)` and `(· / c)` are measurable functions.
For a typeclass assuming measurability of `uncurry (· / ·)` see `MeasurableDiv₂`. -/
@[to_additive]
class MeasurableDiv (G₀ : Type*) [MeasurableSpace G₀] [Div G₀] : Prop where
  measurable_const_div : ∀ c : G₀, Measurable (c / ·)
  measurable_div_const : ∀ c : G₀, Measurable (· / c)


/-- We say that a type has `MeasurableDiv₂` if `uncurry (· / ·)` is a measurable functions.
For a typeclass assuming measurability of `(c / ·)` and `(· / c)` see `MeasurableDiv`. -/
@[to_additive MeasurableSub₂]
class MeasurableDiv₂ (G₀ : Type*) [MeasurableSpace G₀] [Div G₀] : Prop where
  measurable_div : Measurable fun p : G₀ × G₀ => p.1 / p.2


@[to_additive (attr := measurability)]
theorem Measurable.const_div [MeasurableDiv G] (hf : Measurable f) (c : G) :
    Measurable fun x => c / f x :=
  (MeasurableDiv.measurable_const_div c).comp hf


@[to_additive (attr := measurability)]
theorem AEMeasurable.const_div [MeasurableDiv G] (hf : AEMeasurable f μ) (c : G) :
    AEMeasurable (fun x => c / f x) μ :=
  (MeasurableDiv.measurable_const_div c).comp_aemeasurable hf


@[to_additive (attr := measurability)]
theorem Measurable.div_const [MeasurableDiv G] (hf : Measurable f) (c : G) :
    Measurable fun x => f x / c :=
  (MeasurableDiv.measurable_div_const c).comp hf


@[to_additive (attr := measurability)]
theorem AEMeasurable.div_const [MeasurableDiv G] (hf : AEMeasurable f μ) (c : G) :
    AEMeasurable (fun x => f x / c) μ :=
  (MeasurableDiv.measurable_div_const c).comp_aemeasurable hf


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem Measurable.div [MeasurableDiv₂ G] (hf : Measurable f) (hg : Measurable g) :
    Measurable fun a => f a / g a :=
  measurable_div.comp (hf.prod_mk hg)


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
lemma Measurable.div' [MeasurableDiv₂ G] {f g : α → β → G} {h : α → β} (hf : Measurable ↿f)
    (hg : Measurable ↿g) (hh : Measurable h) : Measurable fun a ↦ (f a / g a) (h a) := by
  /-
    G : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝² : MeasurableSpace G
    inst✝¹ : Div G
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableDiv₂ G
    f g : α → β → G
    h : α → β
    hf : Measurable (Function.HasUncurry.uncurry f)
    hg : Measurable (Function.HasUncurry.uncurry g)
    hh : Measurable h
    ⊢ Measurable fun a => HDiv.hDiv (f a) (g a) (h a)
  -/
  simp; fun_prop
        /-
          🎉 no goals
        -/


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem AEMeasurable.div' [MeasurableDiv₂ G] (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (f / g) μ :=
  measurable_div.comp_aemeasurable (hf.prod_mk hg)


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem AEMeasurable.div [MeasurableDiv₂ G] (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    AEMeasurable (fun a => f a / g a) μ :=
  measurable_div.comp_aemeasurable (hf.prod_mk hg)


@[to_additive]
instance (priority := 100) MeasurableDiv₂.toMeasurableDiv [MeasurableDiv₂ G] :
    MeasurableDiv G :=
  ⟨fun _ => measurable_const.div measurable_id, fun _ => measurable_id.div measurable_const⟩


@[to_additive]
instance Pi.measurableDiv {ι : Type*} {α : ι → Type*} [∀ i, Div (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableDiv (α i)] : MeasurableDiv (∀ i, α i) :=
  ⟨fun _ => measurable_pi_iff.mpr fun i => (measurable_pi_apply i).const_div _, fun _ =>
    measurable_pi_iff.mpr fun i => (measurable_pi_apply i).div_const _⟩


@[to_additive Pi.measurableSub₂]
instance Pi.measurableDiv₂ {ι : Type*} {α : ι → Type*} [∀ i, Div (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableDiv₂ (α i)] : MeasurableDiv₂ (∀ i, α i) :=
  ⟨measurable_pi_iff.mpr fun _ => measurable_fst.eval.div measurable_snd.eval⟩


@[measurability]
theorem measurableSet_eq_fun {m : MeasurableSpace α} {E} [MeasurableSpace E] [AddGroup E]
    [MeasurableSingletonClass E] [MeasurableSub₂ E] {f g : α → E} (hf : Measurable f)
    (hg : Measurable g) : MeasurableSet { x | f x = g x } := by
  suffices h_set_eq : { x : α | f x = g x } = { x | (f - g) x = (0 : E) } by
    rw [h_set_eq]
    exact (hf.sub hg) measurableSet_eq
  /-
    α : Type u_3
    m : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (setOf fun x => Eq (f x) (g x)) (setOf fun x => Eq (HSub.hSub f g x) 0)
  -/
  ext
  /-
    case h
    α : Type u_3
    m : MeasurableSpace α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    x✝ : α
    ⊢ Iff (Membership.mem (setOf fun x => Eq (f x) (g x)) x✝) (Membership.mem (set …
  -/
  simp_rw [Set.mem_setOf_eq, Pi.sub_apply, sub_eq_zero]
  /-
    🎉 no goals
  -/


@[measurability]
lemma measurableSet_eq_fun' {β : Type*} [CanonicallyOrderedAddCommMonoid β] [Sub β] [OrderedSub β]
    {_ : MeasurableSpace β} [MeasurableSub₂ β] [MeasurableSingletonClass β]
    {f g : α → β} (hf : Measurable f) (hg : Measurable g) :
    MeasurableSet {x | f x = g x} := by
  have : {a | f a = g a} = {a | (f - g) a = 0} ∩ {a | (g - f) a = 0} := by
    ext
    simp only [Set.mem_setOf_eq, Pi.sub_apply, tsub_eq_zero_iff_le, Set.mem_inter_iff]
    exact ⟨fun h ↦ ⟨h.le, h.symm.le⟩, fun h ↦ le_antisymm h.1 h.2⟩
  /-
    α : Type u_3
    m : MeasurableSpace α
    β : Type u_5
    inst✝⁴ : CanonicallyOrderedAddCommMonoid β
    inst✝³ : Sub β
    inst✝² : OrderedSub β
    x✝ : MeasurableSpace β
    inst✝¹ : MeasurableSub₂ β
    inst✝ : MeasurableSingletonClass β
    f g : α → β
    hf : Measurable f
    hg : Measurable g
    this : Eq (setOf fun a => Eq (f a) (g a)) (Inter.inter (setOf fun a => Eq (HSu …
    ⊢ MeasurableSet (setOf fun x => Eq (f x) (g x))
  -/
  rw [this]
  /-
    α : Type u_3
    m : MeasurableSpace α
    β : Type u_5
    inst✝⁴ : CanonicallyOrderedAddCommMonoid β
    inst✝³ : Sub β
    inst✝² : OrderedSub β
    x✝ : MeasurableSpace β
    inst✝¹ : MeasurableSub₂ β
    inst✝ : MeasurableSingletonClass β
    f g : α → β
    hf : Measurable f
    hg : Measurable g
    this : Eq (setOf fun a => Eq (f a) (g a)) (Inter.inter (setOf fun a => Eq (HSu …
    ⊢ MeasurableSet (Inter.inter (setOf fun a => Eq (HSub.hSub f g a) 0) (setOf fu …
  -/
  exact ((hf.sub hg) (measurableSet_singleton 0)).inter ((hg.sub hf) (measurableSet_singleton 0))
  /-
    🎉 no goals
  -/


theorem nullMeasurableSet_eq_fun {E} [MeasurableSpace E] [AddGroup E] [MeasurableSingletonClass E]
    [MeasurableSub₂ E] {f g : α → E} (hf : AEMeasurable f μ) (hg : AEMeasurable g μ) :
    NullMeasurableSet { x | f x = g x } μ := by
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ MeasureTheory.NullMeasurableSet (setOf fun x => Eq (f x) (g x)) μ
  -/
  apply (measurableSet_eq_fun hf.measurable_mk hg.measurable_mk).nullMeasurableSet.congr
  /-
    α : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun x => Eq (AEMeasurable.mk f hf x …
  -/
  filter_upwards [hf.ae_eq_mk, hg.ae_eq_mk] with x hfx hgx
  /-
    case h
    α : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    x : α
    hfx : Eq (f x) (AEMeasurable.mk f hf x)
    hgx : Eq (g x) (AEMeasurable.mk g hg x)
    ⊢ Eq (setOf (fun x => Eq (AEMeasurable.mk f hf x) (AEMeasurable.mk g hg x)) x) …
  -/
  change (hf.mk f x = hg.mk g x) = (f x = g x)
  /-
    case h
    α : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    f g : α → E
    hf : AEMeasurable f μ
    hg : AEMeasurable g μ
    x : α
    hfx : Eq (f x) (AEMeasurable.mk f hf x)
    hgx : Eq (g x) (AEMeasurable.mk g hg x)
    ⊢ Eq (Eq (AEMeasurable.mk f hf x) (AEMeasurable.mk g hg x)) (Eq (f x) (g x))
  -/
  simp only [hfx, hgx]
  /-
    🎉 no goals
  -/


theorem measurableSet_eq_fun_of_countable {m : MeasurableSpace α} {E} [MeasurableSpace E]
    [MeasurableSingletonClass E] [Countable E] {f g : α → E} (hf : Measurable f)
    (hg : Measurable g) : MeasurableSet { x | f x = g x } := by
  have : { x | f x = g x } = ⋃ j, { x | f x = j } ∩ { x | g x = j } := by
    ext1 x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, exists_eq_right']
  /-
    α : Type u_3
    m : MeasurableSpace α
    E : Type u_5
    inst✝² : MeasurableSpace E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : Countable E
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    this : Eq (setOf fun x => Eq (f x) (g x)) (Set.iUnion fun j => Inter.inter (se …
    ⊢ MeasurableSet (setOf fun x => Eq (f x) (g x))
  -/
  rw [this]
  /-
    α : Type u_3
    m : MeasurableSpace α
    E : Type u_5
    inst✝² : MeasurableSpace E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : Countable E
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    this : Eq (setOf fun x => Eq (f x) (g x)) (Set.iUnion fun j => Inter.inter (se …
    ⊢ MeasurableSet (Set.iUnion fun j => Inter.inter (setOf fun x => Eq (f x) j) ( …
  -/
  refine MeasurableSet.iUnion fun j => MeasurableSet.inter ?_ ?_
    /-
      case refine_1
      α : Type u_3
      m : MeasurableSpace α
      E : Type u_5
      inst✝² : MeasurableSpace E
      inst✝¹ : MeasurableSingletonClass E
      inst✝ : Countable E
      f g : α → E
      hf : Measurable f
      hg : Measurable g
      this : Eq (setOf fun x => Eq (f x) (g x)) (Set.iUnion fun j => Inter.inter (se …
      j : E
      ⊢ MeasurableSet (setOf fun x => Eq (f x) j)
    -/
  · exact hf (measurableSet_singleton j)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      m : MeasurableSpace α
      E : Type u_5
      inst✝² : MeasurableSpace E
      inst✝¹ : MeasurableSingletonClass E
      inst✝ : Countable E
      f g : α → E
      hf : Measurable f
      hg : Measurable g
      this : Eq (setOf fun x => Eq (f x) (g x)) (Set.iUnion fun j => Inter.inter (se …
      j : E
      ⊢ MeasurableSet (setOf fun x => Eq (g x) j)
    -/
  · exact hg (measurableSet_singleton j)
    /-
      🎉 no goals
    -/


theorem ae_eq_trim_of_measurable {α E} {m m0 : MeasurableSpace α} {μ : Measure α}
    [MeasurableSpace E] [AddGroup E] [MeasurableSingletonClass E] [MeasurableSub₂ E]
    (hm : m ≤ m0) {f g : α → E} (hf : Measurable[m] f) (hg : Measurable[m] g) (hfg : f =ᵐ[μ] g) :
    f =ᵐ[μ.trim hm] g := by
  /-
    α : Type u_5
    E : Type u_6
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    hm : LE.le m m0
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyEq f g
  -/
  rwa [Filter.EventuallyEq, ae_iff, trim_measurableSet_eq hm _]
  /-
    α : Type u_5
    E : Type u_6
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace E
    inst✝² : AddGroup E
    inst✝¹ : MeasurableSingletonClass E
    inst✝ : MeasurableSub₂ E
    hm : LE.le m m0
    f g : α → E
    hf : Measurable f
    hg : Measurable g
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ MeasurableSet (setOf fun a => Not (Eq (f a) (g a)))
  -/
  exact @MeasurableSet.compl α _ m (@measurableSet_eq_fun α m E _ _ _ _ _ _ hf hg)
  /-
    🎉 no goals
  -/


/-- We say that a type has `MeasurableNeg` if `x ↦ -x` is a measurable function. -/
class MeasurableNeg (G : Type*) [Neg G] [MeasurableSpace G] : Prop where
  measurable_neg : Measurable (Neg.neg : G → G)


/-- We say that a type has `MeasurableInv` if `x ↦ x⁻¹` is a measurable function. -/
@[to_additive]
class MeasurableInv (G : Type*) [Inv G] [MeasurableSpace G] : Prop where
  measurable_inv : Measurable (Inv.inv : G → G)


@[to_additive]
instance (priority := 100) measurableDiv_of_mul_inv (G : Type*) [MeasurableSpace G]
    [DivInvMonoid G] [MeasurableMul G] [MeasurableInv G] : MeasurableDiv G where
  measurable_const_div c := by
    /-
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c : G
      ⊢ Measurable fun x => HDiv.hDiv c x
    -/
    convert measurable_inv.const_mul c using 1
    /-
      case h.e'_5
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c : G
      ⊢ Eq (fun x => HDiv.hDiv c x) fun x => HMul.hMul c (Inv.inv x)
    -/
    ext1
    /-
      case h.e'_5.h
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c x✝ : G
      ⊢ Eq (HDiv.hDiv c x✝) (HMul.hMul c (Inv.inv x✝))
    -/
    apply div_eq_mul_inv
    /-
      🎉 no goals
    -/
  measurable_div_const c := by
    /-
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c : G
      ⊢ Measurable fun x => HDiv.hDiv x c
    -/
    convert measurable_id.mul_const c⁻¹ using 1
    /-
      case h.e'_5
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c : G
      ⊢ Eq (fun x => HDiv.hDiv x c) fun x => HMul.hMul (id x) (Inv.inv c)
    -/
    ext1
    /-
      case h.e'_5.h
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul G
      inst✝ : MeasurableInv G
      c x✝ : G
      ⊢ Eq (HDiv.hDiv x✝ c) (HMul.hMul (id x✝) (Inv.inv c))
    -/
    apply div_eq_mul_inv
    /-
      🎉 no goals
    -/


@[to_additive (attr := fun_prop, measurability)]
theorem Measurable.inv (hf : Measurable f) : Measurable fun x => (f x)⁻¹ :=
  measurable_inv.comp hf


@[to_additive (attr := fun_prop, measurability)]
theorem AEMeasurable.inv (hf : AEMeasurable f μ) : AEMeasurable (fun x => (f x)⁻¹) μ :=
  measurable_inv.comp_aemeasurable hf


@[to_additive (attr := simp)]
theorem measurable_inv_iff {G : Type*} [Group G] [MeasurableSpace G] [MeasurableInv G]
    {f : α → G} : (Measurable fun x => (f x)⁻¹) ↔ Measurable f :=
               /-
                 α : Type u_3
                 m : MeasurableSpace α
                 G : Type u_4
                 inst✝² : Group G
                 inst✝¹ : MeasurableSpace G
                 inst✝ : MeasurableInv G
                 f : α → G
                 h : Measurable fun x => Inv.inv (f x)
                 ⊢ Measurable f
               -/
  ⟨fun h => by simpa only [inv_inv] using h.inv, fun h => h.inv⟩
               /-
                 🎉 no goals
               -/


@[to_additive (attr := simp)]
theorem aemeasurable_inv_iff {G : Type*} [Group G] [MeasurableSpace G] [MeasurableInv G]
    {f : α → G} : AEMeasurable (fun x => (f x)⁻¹) μ ↔ AEMeasurable f μ :=
               /-
                 α : Type u_3
                 m : MeasurableSpace α
                 μ : MeasureTheory.Measure α
                 G : Type u_4
                 inst✝² : Group G
                 inst✝¹ : MeasurableSpace G
                 inst✝ : MeasurableInv G
                 f : α → G
                 h : AEMeasurable (fun x => Inv.inv (f x)) μ
                 ⊢ AEMeasurable f μ
               -/
  ⟨fun h => by simpa only [inv_inv] using h.inv, fun h => h.inv⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem measurable_inv_iff₀ {G₀ : Type*} [GroupWithZero G₀] [MeasurableSpace G₀]
    [MeasurableInv G₀] {f : α → G₀} : (Measurable fun x => (f x)⁻¹) ↔ Measurable f :=
               /-
                 α : Type u_3
                 m : MeasurableSpace α
                 G₀ : Type u_4
                 inst✝² : GroupWithZero G₀
                 inst✝¹ : MeasurableSpace G₀
                 inst✝ : MeasurableInv G₀
                 f : α → G₀
                 h : Measurable fun x => Inv.inv (f x)
                 ⊢ Measurable f
               -/
  ⟨fun h => by simpa only [inv_inv] using h.inv, fun h => h.inv⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem aemeasurable_inv_iff₀ {G₀ : Type*} [GroupWithZero G₀] [MeasurableSpace G₀]
    [MeasurableInv G₀] {f : α → G₀} : AEMeasurable (fun x => (f x)⁻¹) μ ↔ AEMeasurable f μ :=
               /-
                 α : Type u_3
                 m : MeasurableSpace α
                 μ : MeasureTheory.Measure α
                 G₀ : Type u_4
                 inst✝² : GroupWithZero G₀
                 inst✝¹ : MeasurableSpace G₀
                 inst✝ : MeasurableInv G₀
                 f : α → G₀
                 h : AEMeasurable (fun x => Inv.inv (f x)) μ
                 ⊢ AEMeasurable f μ
               -/
  ⟨fun h => by simpa only [inv_inv] using h.inv, fun h => h.inv⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
instance Pi.measurableInv {ι : Type*} {α : ι → Type*} [∀ i, Inv (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableInv (α i)] : MeasurableInv (∀ i, α i) :=
  ⟨measurable_pi_iff.mpr fun i => (measurable_pi_apply i).inv⟩


@[to_additive]
theorem MeasurableSet.inv {s : Set G} (hs : MeasurableSet s) : MeasurableSet s⁻¹ :=
  measurable_inv hs


@[to_additive]
theorem measurableEmbedding_inv [InvolutiveInv α] [MeasurableInv α] :
    MeasurableEmbedding (Inv.inv (α := α)) :=
  ⟨inv_injective, measurable_inv, fun s hs ↦ s.image_inv_eq_inv ▸ hs.inv⟩


@[to_additive]
theorem Measurable.mul_iff_right {G : Type*} [MeasurableSpace G] [MeasurableSpace α] [CommGroup G]
    [MeasurableMul₂ G] [MeasurableInv G] {f g : α → G} (hf : Measurable f) :
    Measurable (f * g) ↔ Measurable g :=
                                   /-
                                     α : Type u_1
                                     G : Type u_2
                                     inst✝⁴ : MeasurableSpace G
                                     inst✝³ : MeasurableSpace α
                                     inst✝² : CommGroup G
                                     inst✝¹ : MeasurableMul₂ G
                                     inst✝ : MeasurableInv G
                                     f g : α → G
                                     hf : Measurable f
                                     h : Measurable (HMul.hMul f g)
                                     ⊢ Eq g (HMul.hMul (HMul.hMul f g) (Inv.inv f))
                                   -/
  ⟨fun h ↦ show g = f * g * f⁻¹ by simp only [mul_inv_cancel_comm] ▸ h.mul hf.inv,
                                   /-
                                     🎉 no goals
                                   -/
    fun h ↦ hf.mul h⟩


@[to_additive]
theorem AEMeasurable.mul_iff_right {G : Type*} [MeasurableSpace G] [MeasurableSpace α] [CommGroup G]
    [MeasurableMul₂ G] [MeasurableInv G] {μ : Measure α} {f g : α → G} (hf : AEMeasurable f μ) :
    AEMeasurable (f * g) μ ↔ AEMeasurable g μ :=
                                   /-
                                     α : Type u_1
                                     G : Type u_2
                                     inst✝⁴ : MeasurableSpace G
                                     inst✝³ : MeasurableSpace α
                                     inst✝² : CommGroup G
                                     inst✝¹ : MeasurableMul₂ G
                                     inst✝ : MeasurableInv G
                                     μ : MeasureTheory.Measure α
                                     f g : α → G
                                     hf : AEMeasurable f μ
                                     h : AEMeasurable (HMul.hMul f g) μ
                                     ⊢ Eq g (HMul.hMul (HMul.hMul f g) (Inv.inv f))
                                   -/
  ⟨fun h ↦ show g = f * g * f⁻¹ by simp only [mul_inv_cancel_comm] ▸ h.mul hf.inv,
                                   /-
                                     🎉 no goals
                                   -/
    fun h ↦ hf.mul h⟩


@[to_additive]
theorem Measurable.mul_iff_left {G : Type*} [MeasurableSpace G] [MeasurableSpace α] [CommGroup G]
    [MeasurableMul₂ G] [MeasurableInv G] {f g : α → G} (hf : Measurable f) :
    Measurable (g * f) ↔ Measurable g :=
  mul_comm g f ▸ Measurable.mul_iff_right hf


@[to_additive]
theorem AEMeasurable.mul_iff_left {G : Type*} [MeasurableSpace G] [MeasurableSpace α] [CommGroup G]
    [MeasurableMul₂ G] [MeasurableInv G] {μ : Measure α} {f g : α → G} (hf : AEMeasurable f μ) :
    AEMeasurable (g * f) μ ↔ AEMeasurable g μ :=
  mul_comm g f ▸ AEMeasurable.mul_iff_right hf


/-- `DivInvMonoid.Pow` is measurable. -/
instance DivInvMonoid.measurableZPow (G : Type u) [DivInvMonoid G] [MeasurableSpace G]
    [MeasurableMul₂ G] [MeasurableInv G] : MeasurablePow G ℤ :=
  ⟨measurable_from_prod_countable fun n => by
      /-
        α : Type u_1
        G : Type u
        inst✝³ : DivInvMonoid G
        inst✝² : MeasurableSpace G
        inst✝¹ : MeasurableMul₂ G
        inst✝ : MeasurableInv G
        n : Int
        ⊢ Measurable fun x => HPow.hPow { fst := x, snd := n }.1 { fst := x, snd := n  …
      -/
      cases' n with n n
        /-
          case ofNat
          α : Type u_1
          G : Type u
          inst✝³ : DivInvMonoid G
          inst✝² : MeasurableSpace G
          inst✝¹ : MeasurableMul₂ G
          inst✝ : MeasurableInv G
          n : Nat
          ⊢ Measurable fun x => HPow.hPow { fst := x, snd := Int.ofNat n }.1 { fst := x, …
        -/
      · simp_rw [Int.ofNat_eq_coe, zpow_natCast]
        /-
          case ofNat
          α : Type u_1
          G : Type u
          inst✝³ : DivInvMonoid G
          inst✝² : MeasurableSpace G
          inst✝¹ : MeasurableMul₂ G
          inst✝ : MeasurableInv G
          n : Nat
          ⊢ Measurable fun x => HPow.hPow x n
        -/
        exact measurable_id.pow_const _
        /-
          🎉 no goals
        -/
        /-
          case negSucc
          α : Type u_1
          G : Type u
          inst✝³ : DivInvMonoid G
          inst✝² : MeasurableSpace G
          inst✝¹ : MeasurableMul₂ G
          inst✝ : MeasurableInv G
          n : Nat
          ⊢ Measurable fun x => HPow.hPow { fst := x, snd := Int.negSucc n }.1 { fst :=  …
        -/
      · simp_rw [zpow_negSucc]
        /-
          case negSucc
          α : Type u_1
          G : Type u
          inst✝³ : DivInvMonoid G
          inst✝² : MeasurableSpace G
          inst✝¹ : MeasurableMul₂ G
          inst✝ : MeasurableInv G
          n : Nat
          ⊢ Measurable fun x => Inv.inv (HPow.hPow x (HAdd.hAdd n 1))
        -/
        exact (measurable_id.pow_const (n + 1)).inv⟩
        /-
          🎉 no goals
        -/


@[to_additive]
instance (priority := 100) measurableDiv₂_of_mul_inv (G : Type*) [MeasurableSpace G]
    [DivInvMonoid G] [MeasurableMul₂ G] [MeasurableInv G] : MeasurableDiv₂ G :=
  ⟨by
    /-
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul₂ G
      inst✝ : MeasurableInv G
      ⊢ Measurable fun p => HDiv.hDiv p.1 p.2
    -/
    simp only [div_eq_mul_inv]
    /-
      α : Type u_1
      G : Type u_2
      inst✝³ : MeasurableSpace G
      inst✝² : DivInvMonoid G
      inst✝¹ : MeasurableMul₂ G
      inst✝ : MeasurableInv G
      ⊢ Measurable fun p => HMul.hMul p.1 (Inv.inv p.2)
    -/
    exact measurable_fst.mul measurable_snd.inv⟩
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) MeasurableDiv.toMeasurableInv [MeasurableSpace α] [Group α]
    [MeasurableDiv α] : MeasurableInv α where
                       /-
                         α : Type u_1
                         inst✝² : MeasurableSpace α
                         inst✝¹ : Group α
                         inst✝ : MeasurableDiv α
                         ⊢ Measurable Inv.inv
                       -/
  measurable_inv := by simpa using measurable_const_div (1 : α)
                       /-
                         🎉 no goals
                       -/


/-- We say that the action of `M` on `α` has `MeasurableVAdd` if for each `c` the map `x ↦ c +ᵥ x`
is a measurable function and for each `x` the map `c ↦ c +ᵥ x` is a measurable function. -/
class MeasurableVAdd (M α : Type*) [VAdd M α] [MeasurableSpace M] [MeasurableSpace α] :
    Prop where
  measurable_const_vadd : ∀ c : M, Measurable (c +ᵥ · : α → α)
  measurable_vadd_const : ∀ x : α, Measurable (· +ᵥ x : M → α)


/-- We say that the action of `M` on `α` has `MeasurableSMul` if for each `c` the map `x ↦ c • x`
is a measurable function and for each `x` the map `c ↦ c • x` is a measurable function. -/
@[to_additive]
class MeasurableSMul (M α : Type*) [SMul M α] [MeasurableSpace M] [MeasurableSpace α] :
    Prop where
  measurable_const_smul : ∀ c : M, Measurable (c • · : α → α)
  measurable_smul_const : ∀ x : α, Measurable (· • x : M → α)


/-- We say that the action of `M` on `α` has `MeasurableVAdd₂` if the map
`(c, x) ↦ c +ᵥ x` is a measurable function. -/
class MeasurableVAdd₂ (M α : Type*) [VAdd M α] [MeasurableSpace M] [MeasurableSpace α] :
    Prop where
  measurable_vadd : Measurable (Function.uncurry (· +ᵥ ·) : M × α → α)


/-- We say that the action of `M` on `α` has `Measurable_SMul₂` if the map
`(c, x) ↦ c • x` is a measurable function. -/
@[to_additive MeasurableVAdd₂]
class MeasurableSMul₂ (M α : Type*) [SMul M α] [MeasurableSpace M] [MeasurableSpace α] :
    Prop where
  measurable_smul : Measurable (Function.uncurry (· • ·) : M × α → α)


@[to_additive]
instance measurableSMul_of_mul (M : Type*) [Mul M] [MeasurableSpace M] [MeasurableMul M] :
    MeasurableSMul M M :=
  ⟨measurable_id.const_mul, measurable_id.mul_const⟩


@[to_additive]
instance measurableSMul₂_of_mul (M : Type*) [Mul M] [MeasurableSpace M] [MeasurableMul₂ M] :
    MeasurableSMul₂ M M :=
  ⟨measurable_mul⟩


@[to_additive]
instance Submonoid.measurableSMul {M α} [MeasurableSpace M] [MeasurableSpace α] [Monoid M]
    [MulAction M α] [MeasurableSMul M α] (s : Submonoid M) : MeasurableSMul s α :=
               /-
                 α✝ : Type u_1
                 M : Type u_2
                 α : Type u_3
                 inst✝⁴ : MeasurableSpace M
                 inst✝³ : MeasurableSpace α
                 inst✝² : Monoid M
                 inst✝¹ : MulAction M α
                 inst✝ : MeasurableSMul M α
                 s : Submonoid M
                 c : Subtype fun x => Membership.mem s x
                 ⊢ Measurable fun x => HSMul.hSMul c x
               -/
  ⟨fun c => by simpa only using measurable_const_smul (c : M), fun x =>
               /-
                 🎉 no goals
               -/
    (measurable_smul_const x : Measurable fun c : M => c • x).comp measurable_subtype_coe⟩


@[to_additive]
instance Subgroup.measurableSMul {G α} [MeasurableSpace G] [MeasurableSpace α] [Group G]
    [MulAction G α] [MeasurableSMul G α] (s : Subgroup G) : MeasurableSMul s α :=
  s.toSubmonoid.measurableSMul


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem Measurable.smul [MeasurableSMul₂ M X] (hf : Measurable f) (hg : Measurable g) :
    Measurable fun x => f x • g x :=
  measurable_smul.comp (hf.prod_mk hg)


/-- Compositional version of `Measurable.smul` for use by `fun_prop`. -/
@[to_additive (attr := fun_prop)
"Compositional version of `Measurable.vadd` for use by `fun_prop`."]
lemma Measurable.smul' [MeasurableSMul₂ M X] {f : α → β → M} {g : α → β → X} {h : α → β}
    (hf : Measurable ↿f) (hg : Measurable ↿g) (hh : Measurable h) :
                                               /-
                                                 M : Type u_2
                                                 X : Type u_3
                                                 α : Type u_4
                                                 β : Type u_5
                                                 inst✝³ : MeasurableSpace M
                                                 inst✝² : MeasurableSpace X
                                                 inst✝¹ : SMul M X
                                                 m : MeasurableSpace α
                                                 mβ : MeasurableSpace β
                                                 inst✝ : MeasurableSMul₂ M X
                                                 f : α → β → M
                                                 g : α → β → X
                                                 h : α → β
                                                 hf : Measurable (Function.HasUncurry.uncurry f)
                                                 hg : Measurable (Function.HasUncurry.uncurry g)
                                                 hh : Measurable h
                                                 ⊢ Measurable fun a => HSMul.hSMul (f a) (g a) (h a)
                                               -/
    Measurable fun a ↦ (f a • g a) (h a) := by simp; fun_prop
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive (attr := fun_prop, aesop safe 20 apply (rule_sets := [Measurable]))]
theorem AEMeasurable.smul [MeasurableSMul₂ M X] {μ : Measure α} (hf : AEMeasurable f μ)
    (hg : AEMeasurable g μ) : AEMeasurable (fun x => f x • g x) μ :=
  MeasurableSMul₂.measurable_smul.comp_aemeasurable (hf.prod_mk hg)


@[to_additive]
instance (priority := 100) MeasurableSMul₂.toMeasurableSMul [MeasurableSMul₂ M X] :
    MeasurableSMul M X :=
  ⟨fun _ => measurable_const.smul measurable_id, fun _ => measurable_id.smul measurable_const⟩


@[to_additive (attr := measurability)]
theorem Measurable.smul_const (hf : Measurable f) (y : X) : Measurable fun x => f x • y :=
  (MeasurableSMul.measurable_smul_const y).comp hf


@[to_additive (attr := measurability)]
theorem AEMeasurable.smul_const (hf : AEMeasurable f μ) (y : X) :
    AEMeasurable (fun x => f x • y) μ :=
  (MeasurableSMul.measurable_smul_const y).comp_aemeasurable hf


@[to_additive (attr := fun_prop, measurability)]
theorem Measurable.const_smul (hg : Measurable g) (c : M) : Measurable (c • g) :=
  (MeasurableSMul.measurable_const_smul c).comp hg


/-- Compositional version of `Measurable.const_smul` for use by `fun_prop`. -/
@[to_additive (attr := fun_prop)
"Compositional version of `Measurable.const_vadd` for use by `fun_prop`."]
lemma Measurable.const_smul' {g : α → β → X} {h : α → β} (hg : Measurable ↿g) (hh : Measurable h)
    (c : M) : Measurable fun a ↦ (c • g a) (h a) :=
  (hg.comp <| measurable_id.prod_mk hh).const_smul _


@[to_additive (attr := measurability)]
theorem AEMeasurable.const_smul' (hg : AEMeasurable g μ) (c : M) :
    AEMeasurable (fun x => c • g x) μ :=
  (MeasurableSMul.measurable_const_smul c).comp_aemeasurable hg


@[to_additive (attr := measurability)]
theorem AEMeasurable.const_smul (hf : AEMeasurable g μ) (c : M) : AEMeasurable (c • g) μ :=
  hf.const_smul' c


@[to_additive]
instance Pi.measurableSMul {ι : Type*} {α : ι → Type*} [∀ i, SMul M (α i)]
    [∀ i, MeasurableSpace (α i)] [∀ i, MeasurableSMul M (α i)] :
    MeasurableSMul M (∀ i, α i) :=
  ⟨fun _ => measurable_pi_iff.mpr fun i => (measurable_pi_apply i).const_smul _, fun _ =>
    measurable_pi_iff.mpr fun _ => measurable_smul_const _⟩


/-- `AddMonoid.SMul` is measurable. -/
instance AddMonoid.measurableSMul_nat₂ (M : Type*) [AddMonoid M] [MeasurableSpace M]
    [MeasurableAdd₂ M] : MeasurableSMul₂ ℕ M :=
  ⟨by
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁶ : MeasurableSpace M✝
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝³ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝² : AddMonoid M
      inst✝¹ : MeasurableSpace M
      inst✝ : MeasurableAdd₂ M
      ⊢ Measurable (Function.uncurry fun x1 x2 => HSMul.hSMul x1 x2)
    -/
    suffices Measurable fun p : M × ℕ => p.2 • p.1 by apply this.comp measurable_swap
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁶ : MeasurableSpace M✝
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝³ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝² : AddMonoid M
      inst✝¹ : MeasurableSpace M
      inst✝ : MeasurableAdd₂ M
      ⊢ Measurable fun p => HSMul.hSMul p.2 p.1
    -/
    refine measurable_from_prod_countable fun n => ?_
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁶ : MeasurableSpace M✝
      inst✝⁵ : MeasurableSpace X
      inst✝⁴ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝³ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝² : AddMonoid M
      inst✝¹ : MeasurableSpace M
      inst✝ : MeasurableAdd₂ M
      n : Nat
      ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := n }.2 { fst := x, snd :=  …
    -/
    induction' n with n ih
      /-
        case zero
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁶ : MeasurableSpace M✝
        inst✝⁵ : MeasurableSpace X
        inst✝⁴ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝³ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝² : AddMonoid M
        inst✝¹ : MeasurableSpace M
        inst✝ : MeasurableAdd₂ M
        ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := 0 }.2 { fst := x, snd :=  …
      -/
    · simp only [zero_smul, ← Pi.zero_def, measurable_zero]
      /-
        🎉 no goals
      -/
      /-
        case succ
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁶ : MeasurableSpace M✝
        inst✝⁵ : MeasurableSpace X
        inst✝⁴ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝³ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝² : AddMonoid M
        inst✝¹ : MeasurableSpace M
        inst✝ : MeasurableAdd₂ M
        n : Nat
        ih : Measurable fun x => HSMul.hSMul { fst := x, snd := n }.2 { fst := x, snd  …
        ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := HAdd.hAdd n 1 }.2 { fst : …
      -/
    · simp only [succ_nsmul]
      /-
        case succ
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁶ : MeasurableSpace M✝
        inst✝⁵ : MeasurableSpace X
        inst✝⁴ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝³ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝² : AddMonoid M
        inst✝¹ : MeasurableSpace M
        inst✝ : MeasurableAdd₂ M
        n : Nat
        ih : Measurable fun x => HSMul.hSMul { fst := x, snd := n }.2 { fst := x, snd  …
        ⊢ Measurable fun x => HAdd.hAdd (HSMul.hSMul n x) x
      -/
      exact ih.add measurable_id⟩
      /-
        🎉 no goals
      -/


/-- `SubNegMonoid.SMulInt` is measurable. -/
instance SubNegMonoid.measurableSMul_int₂ (M : Type*) [SubNegMonoid M] [MeasurableSpace M]
    [MeasurableAdd₂ M] [MeasurableNeg M] : MeasurableSMul₂ ℤ M :=
  ⟨by
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁷ : MeasurableSpace M✝
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝⁴ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝³ : SubNegMonoid M
      inst✝² : MeasurableSpace M
      inst✝¹ : MeasurableAdd₂ M
      inst✝ : MeasurableNeg M
      ⊢ Measurable (Function.uncurry fun x1 x2 => HSMul.hSMul x1 x2)
    -/
    suffices Measurable fun p : M × ℤ => p.2 • p.1 by apply this.comp measurable_swap
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁷ : MeasurableSpace M✝
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝⁴ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝³ : SubNegMonoid M
      inst✝² : MeasurableSpace M
      inst✝¹ : MeasurableAdd₂ M
      inst✝ : MeasurableNeg M
      ⊢ Measurable fun p => HSMul.hSMul p.2 p.1
    -/
    refine measurable_from_prod_countable fun n => ?_
    /-
      α✝ : Type u_1
      M✝ : Type u_2
      X : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝⁷ : MeasurableSpace M✝
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : SMul M✝ X
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → M✝
      g : α → X
      inst✝⁴ : MeasurableSMul M✝ X
      μ : MeasureTheory.Measure α
      M : Type u_6
      inst✝³ : SubNegMonoid M
      inst✝² : MeasurableSpace M
      inst✝¹ : MeasurableAdd₂ M
      inst✝ : MeasurableNeg M
      n : Int
      ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := n }.2 { fst := x, snd :=  …
    -/
    induction' n with n n ih
      /-
        case ofNat
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁷ : MeasurableSpace M✝
        inst✝⁶ : MeasurableSpace X
        inst✝⁵ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝⁴ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝³ : SubNegMonoid M
        inst✝² : MeasurableSpace M
        inst✝¹ : MeasurableAdd₂ M
        inst✝ : MeasurableNeg M
        n : Nat
        ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := Int.ofNat n }.2 { fst :=  …
      -/
    · simp only [Int.ofNat_eq_coe, natCast_zsmul]
      /-
        case ofNat
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁷ : MeasurableSpace M✝
        inst✝⁶ : MeasurableSpace X
        inst✝⁵ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝⁴ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝³ : SubNegMonoid M
        inst✝² : MeasurableSpace M
        inst✝¹ : MeasurableAdd₂ M
        inst✝ : MeasurableNeg M
        n : Nat
        ⊢ Measurable fun x => HSMul.hSMul n x
      -/
      exact measurable_const_smul _
      /-
        🎉 no goals
      -/
      /-
        case negSucc
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁷ : MeasurableSpace M✝
        inst✝⁶ : MeasurableSpace X
        inst✝⁵ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝⁴ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝³ : SubNegMonoid M
        inst✝² : MeasurableSpace M
        inst✝¹ : MeasurableAdd₂ M
        inst✝ : MeasurableNeg M
        n : Nat
        ⊢ Measurable fun x => HSMul.hSMul { fst := x, snd := Int.negSucc n }.2 { fst : …
      -/
    · simp only [negSucc_zsmul]
      /-
        case negSucc
        α✝ : Type u_1
        M✝ : Type u_2
        X : Type u_3
        α : Type u_4
        β : Type u_5
        inst✝⁷ : MeasurableSpace M✝
        inst✝⁶ : MeasurableSpace X
        inst✝⁵ : SMul M✝ X
        m : MeasurableSpace α
        mβ : MeasurableSpace β
        f : α → M✝
        g : α → X
        inst✝⁴ : MeasurableSMul M✝ X
        μ : MeasureTheory.Measure α
        M : Type u_6
        inst✝³ : SubNegMonoid M
        inst✝² : MeasurableSpace M
        inst✝¹ : MeasurableAdd₂ M
        inst✝ : MeasurableNeg M
        n : Nat
        ⊢ Measurable fun x => Neg.neg (HSMul.hSMul (HAdd.hAdd n 1) x)
      -/
      exact (measurable_const_smul _).neg⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem Measurable.measurableSMul₂_iterateMulAct (h : Measurable f) :
    MeasurableSMul₂ (IterateMulAct f) α where
  measurable_smul :=
    suffices Measurable fun p : α × IterateMulAct f ↦ f^[p.2.val] p.1 from this.comp measurable_swap
    measurable_from_prod_countable fun n ↦ h.iterate n.val


@[to_additive (attr := simp)]
theorem measurableSMul_iterateMulAct : MeasurableSMul (IterateMulAct f) α ↔ Measurable f :=
  ⟨fun _ ↦ measurable_const_smul (IterateMulAct.mk (f := f) 1), fun h ↦
    have := h.measurableSMul₂_iterateMulAct; inferInstance⟩


@[to_additive (attr := simp)]
theorem measurableSMul₂_iterateMulAct : MeasurableSMul₂ (IterateMulAct f) α ↔ Measurable f :=
  ⟨fun _ ↦ measurableSMul_iterateMulAct.mp inferInstance,
    Measurable.measurableSMul₂_iterateMulAct⟩


@[to_additive]
theorem measurable_const_smul_iff (c : G) : (Measurable fun x => c • f x) ↔ Measurable f :=
               /-
                 β : Type u_3
                 α : Type u_4
                 inst✝⁵ : MeasurableSpace β
                 inst✝⁴ : MeasurableSpace α
                 f : α → β
                 G : Type u_5
                 inst✝³ : Group G
                 inst✝² : MeasurableSpace G
                 inst✝¹ : MulAction G β
                 inst✝ : MeasurableSMul G β
                 c : G
                 h : Measurable fun x => HSMul.hSMul c (f x)
                 ⊢ Measurable f
               -/
  ⟨fun h => by simpa [inv_smul_smul, Pi.smul_def] using h.const_smul c⁻¹, fun h => h.const_smul c⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
theorem aemeasurable_const_smul_iff (c : G) :
    AEMeasurable (fun x => c • f x) μ ↔ AEMeasurable f μ :=
               /-
                 β : Type u_3
                 α : Type u_4
                 inst✝⁵ : MeasurableSpace β
                 inst✝⁴ : MeasurableSpace α
                 f : α → β
                 μ : MeasureTheory.Measure α
                 G : Type u_5
                 inst✝³ : Group G
                 inst✝² : MeasurableSpace G
                 inst✝¹ : MulAction G β
                 inst✝ : MeasurableSMul G β
                 c : G
                 h : AEMeasurable (fun x => HSMul.hSMul c (f x)) μ
                 ⊢ AEMeasurable f μ
               -/
  ⟨fun h => by simpa [inv_smul_smul, Pi.smul_def] using h.const_smul c⁻¹, fun h => h.const_smul c⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
instance Units.instMeasurableSpace : MeasurableSpace Mˣ := MeasurableSpace.comap ((↑) : Mˣ → M) ‹_›


@[to_additive]
instance Units.measurableSMul : MeasurableSMul Mˣ β where
  measurable_const_smul c := (measurable_const_smul (c : M) : _)
  measurable_smul_const x :=
    (measurable_smul_const x : Measurable fun c : M => c • x).comp MeasurableSpace.le_map_comap


@[to_additive]
nonrec theorem IsUnit.measurable_const_smul_iff {c : M} (hc : IsUnit c) :
    (Measurable fun x => c • f x) ↔ Measurable f :=
  let ⟨u, hu⟩ := hc
  hu ▸ measurable_const_smul_iff u


@[to_additive]
nonrec theorem IsUnit.aemeasurable_const_smul_iff {c : M} (hc : IsUnit c) :
    AEMeasurable (fun x => c • f x) μ ↔ AEMeasurable f μ :=
  let ⟨u, hu⟩ := hc
  hu ▸ aemeasurable_const_smul_iff u


theorem measurable_const_smul_iff₀ {c : G₀} (hc : c ≠ 0) :
    (Measurable fun x => c • f x) ↔ Measurable f :=
  (IsUnit.mk0 c hc).measurable_const_smul_iff


theorem aemeasurable_const_smul_iff₀ {c : G₀} (hc : c ≠ 0) :
    AEMeasurable (fun x => c • f x) μ ↔ AEMeasurable f μ :=
  (IsUnit.mk0 c hc).aemeasurable_const_smul_iff


@[to_additive]
instance MulOpposite.instMeasurableSpace {α : Type*} [h : MeasurableSpace α] :
    MeasurableSpace αᵐᵒᵖ :=
  MeasurableSpace.map op h


@[to_additive]
theorem measurable_mul_op {α : Type*} [MeasurableSpace α] : Measurable (op : α → αᵐᵒᵖ) := fun _ =>
  id


@[to_additive]
theorem measurable_mul_unop {α : Type*} [MeasurableSpace α] : Measurable (unop : αᵐᵒᵖ → α) :=
  fun _ => id


@[to_additive]
instance MulOpposite.instMeasurableMul {M : Type*} [Mul M] [MeasurableSpace M]
    [MeasurableMul M] : MeasurableMul Mᵐᵒᵖ :=
  ⟨fun _ => measurable_mul_op.comp (measurable_mul_unop.mul_const _), fun _ =>
    measurable_mul_op.comp (measurable_mul_unop.const_mul _)⟩


@[to_additive]
instance MulOpposite.instMeasurableMul₂ {M : Type*} [Mul M] [MeasurableSpace M]
    [MeasurableMul₂ M] : MeasurableMul₂ Mᵐᵒᵖ :=
  ⟨measurable_mul_op.comp
      ((measurable_mul_unop.comp measurable_snd).mul (measurable_mul_unop.comp measurable_fst))⟩


/-- If a scalar is central, then its right action is measurable when its left action is. -/
nonrec instance MeasurableSMul.op {M α} [MeasurableSpace M] [MeasurableSpace α] [SMul M α]
    [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] [MeasurableSMul M α] : MeasurableSMul Mᵐᵒᵖ α :=
  ⟨MulOpposite.rec' fun c =>
      show Measurable fun x => op c • x by
        /-
          α✝ : Type u_1
          M : Type u_2
          α : Type u_3
          inst✝⁵ : MeasurableSpace M
          inst✝⁴ : MeasurableSpace α
          inst✝³ : SMul M α
          inst✝² : SMul (MulOpposite M) α
          inst✝¹ : IsCentralScalar M α
          inst✝ : MeasurableSMul M α
          c : M
          ⊢ Measurable fun x => HSMul.hSMul (MulOpposite.op c) x
        -/
        simpa only [op_smul_eq_smul] using measurable_const_smul c,
        /-
          🎉 no goals
        -/
    fun x =>
    show Measurable fun c => op (unop c) • x by
      /-
        α✝ : Type u_1
        M : Type u_2
        α : Type u_3
        inst✝⁵ : MeasurableSpace M
        inst✝⁴ : MeasurableSpace α
        inst✝³ : SMul M α
        inst✝² : SMul (MulOpposite M) α
        inst✝¹ : IsCentralScalar M α
        inst✝ : MeasurableSMul M α
        x : α
        ⊢ Measurable fun c => HSMul.hSMul (MulOpposite.op (MulOpposite.unop c)) x
      -/
      simpa only [op_smul_eq_smul] using (measurable_smul_const x).comp measurable_mul_unop⟩
      /-
        🎉 no goals
      -/


/-- If a scalar is central, then its right action is measurable when its left action is. -/
nonrec instance MeasurableSMul₂.op {M α} [MeasurableSpace M] [MeasurableSpace α] [SMul M α]
    [SMul Mᵐᵒᵖ α] [IsCentralScalar M α] [MeasurableSMul₂ M α] : MeasurableSMul₂ Mᵐᵒᵖ α :=
  ⟨show Measurable fun x : Mᵐᵒᵖ × α => op (unop x.1) • x.2 by
      /-
        α✝ : Type u_1
        M : Type u_2
        α : Type u_3
        inst✝⁵ : MeasurableSpace M
        inst✝⁴ : MeasurableSpace α
        inst✝³ : SMul M α
        inst✝² : SMul (MulOpposite M) α
        inst✝¹ : IsCentralScalar M α
        inst✝ : MeasurableSMul₂ M α
        ⊢ Measurable fun x => HSMul.hSMul (MulOpposite.op (MulOpposite.unop x.1)) x.2
      -/
      simp_rw [op_smul_eq_smul]
      /-
        α✝ : Type u_1
        M : Type u_2
        α : Type u_3
        inst✝⁵ : MeasurableSpace M
        inst✝⁴ : MeasurableSpace α
        inst✝³ : SMul M α
        inst✝² : SMul (MulOpposite M) α
        inst✝¹ : IsCentralScalar M α
        inst✝ : MeasurableSMul₂ M α
        ⊢ Measurable fun x => HSMul.hSMul (MulOpposite.unop x.1) x.2
      -/
      exact (measurable_mul_unop.comp measurable_fst).smul measurable_snd⟩
      /-
        🎉 no goals
      -/


@[to_additive]
instance measurableSMul_opposite_of_mul {M : Type*} [Mul M] [MeasurableSpace M]
    [MeasurableMul M] : MeasurableSMul Mᵐᵒᵖ M :=
  ⟨fun c => measurable_mul_const (unop c), fun x => measurable_mul_unop.const_mul x⟩


@[to_additive]
instance measurableSMul₂_opposite_of_mul {M : Type*} [Mul M] [MeasurableSpace M]
    [MeasurableMul₂ M] : MeasurableSMul₂ Mᵐᵒᵖ M :=
  ⟨measurable_snd.mul (measurable_mul_unop.comp measurable_fst)⟩


@[to_additive (attr := measurability)]
theorem List.measurable_prod' (l : List (α → M)) (hl : ∀ f ∈ l, Measurable f) :
    Measurable l.prod := by
  /-
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → Measurable f
    ⊢ Measurable l.prod
  -/
  induction' l with f l ihl; · exact measurable_one
                               /-
                                 🎉 no goals
                               -/
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → Measurable f) → Measurable l.prod
    hl : ∀ (f_1 : α → M), Membership.mem (List.cons f l) f_1 → Measurable f_1
    ⊢ Measurable (List.cons f l).prod
  -/
  rw [List.forall_mem_cons] at hl
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → Measurable f) → Measurable l.prod
    hl : And (Measurable f) (∀ (x : α → M), Membership.mem l x → Measurable x)
    ⊢ Measurable (List.cons f l).prod
  -/
  rw [List.prod_cons]
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → Measurable f) → Measurable l.prod
    hl : And (Measurable f) (∀ (x : α → M), Membership.mem l x → Measurable x)
    ⊢ Measurable (HMul.hMul f l.prod)
  -/
  exact hl.1.mul (ihl hl.2)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem List.aemeasurable_prod' (l : List (α → M)) (hl : ∀ f ∈ l, AEMeasurable f μ) :
    AEMeasurable l.prod μ := by
  /-
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → AEMeasurable f μ
    ⊢ AEMeasurable l.prod μ
  -/
  induction' l with f l ihl; · exact aemeasurable_one
                               /-
                                 🎉 no goals
                               -/
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → AEMeasurable f μ) → AEMeasurable l. …
    hl : ∀ (f_1 : α → M), Membership.mem (List.cons f l) f_1 → AEMeasurable f_1 μ
    ⊢ AEMeasurable (List.cons f l).prod μ
  -/
  rw [List.forall_mem_cons] at hl
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → AEMeasurable f μ) → AEMeasurable l. …
    hl : And (AEMeasurable f μ) (∀ (x : α → M), Membership.mem l x → AEMeasurable  …
    ⊢ AEMeasurable (List.cons f l).prod μ
  -/
  rw [List.prod_cons]
  /-
    case cons
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → AEMeasurable f μ) → AEMeasurable l. …
    hl : And (AEMeasurable f μ) (∀ (x : α → M), Membership.mem l x → AEMeasurable  …
    ⊢ AEMeasurable (HMul.hMul f l.prod) μ
  -/
  exact hl.1.mul (ihl hl.2)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem List.measurable_prod (l : List (α → M)) (hl : ∀ f ∈ l, Measurable f) :
    Measurable fun x => (l.map fun f : α → M => f x).prod := by
  /-
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → Measurable f
    ⊢ Measurable fun x => (List.map (fun f => f x) l).prod
  -/
  simpa only [← Pi.list_prod_apply] using l.measurable_prod' hl
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem List.aemeasurable_prod (l : List (α → M)) (hl : ∀ f ∈ l, AEMeasurable f μ) :
    AEMeasurable (fun x => (l.map fun f : α → M => f x).prod) μ := by
  /-
    M : Type u_2
    α : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → AEMeasurable f μ
    ⊢ AEMeasurable (fun x => (List.map (fun f => f x) l).prod) μ
  -/
  simpa only [← Pi.list_prod_apply] using l.aemeasurable_prod' hl
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem Multiset.measurable_prod' (l : Multiset (α → M)) (hl : ∀ f ∈ l, Measurable f) :
    Measurable l.prod := by
  /-
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    l : Multiset (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → Measurable f
    ⊢ Measurable l.prod
  -/
  rcases l with ⟨l⟩
  /-
    case mk
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    l✝ : Multiset (α → M)
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem (Quot.mk (⇑(List.isSetoid (α → M))) l) f →  …
    ⊢ Measurable (Multiset.prod (Quot.mk (⇑(List.isSetoid (α → M))) l))
  -/
  simpa using l.measurable_prod' (by simpa using hl)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem Multiset.aemeasurable_prod' (l : Multiset (α → M)) (hl : ∀ f ∈ l, AEMeasurable f μ) :
    AEMeasurable l.prod μ := by
  /-
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l : Multiset (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → AEMeasurable f μ
    ⊢ AEMeasurable l.prod μ
  -/
  rcases l with ⟨l⟩
  /-
    case mk
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    l✝ : Multiset (α → M)
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem (Quot.mk (⇑(List.isSetoid (α → M))) l) f →  …
    ⊢ AEMeasurable (Multiset.prod (Quot.mk (⇑(List.isSetoid (α → M))) l)) μ
  -/
  simpa using l.aemeasurable_prod' (by simpa using hl)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem Multiset.measurable_prod (s : Multiset (α → M)) (hs : ∀ f ∈ s, Measurable f) :
    Measurable fun x => (s.map fun f : α → M => f x).prod := by
  /-
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    s : Multiset (α → M)
    hs : ∀ (f : α → M), Membership.mem s f → Measurable f
    ⊢ Measurable fun x => (Multiset.map (fun f => f x) s).prod
  -/
  simpa only [← Pi.multiset_prod_apply] using s.measurable_prod' hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem Multiset.aemeasurable_prod (s : Multiset (α → M)) (hs : ∀ f ∈ s, AEMeasurable f μ) :
    AEMeasurable (fun x => (s.map fun f : α → M => f x).prod) μ := by
  /-
    M : Type u_2
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Multiset (α → M)
    hs : ∀ (f : α → M), Membership.mem s f → AEMeasurable f μ
    ⊢ AEMeasurable (fun x => (Multiset.map (fun f => f x) s).prod) μ
  -/
  simpa only [← Pi.multiset_prod_apply] using s.aemeasurable_prod' hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := fun_prop, measurability)]
theorem Finset.measurable_prod (s : Finset ι) (hf : ∀ i ∈ s, Measurable (f i)) :
    Measurable fun a ↦ ∏ i ∈ s, f i a := by
  /-
    M : Type u_2
    ι : Type u_3
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    f : ι → α → M
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    ⊢ Measurable fun a => s.prod fun i => f i a
  -/
  simp_rw [← Finset.prod_apply]
  /-
    M : Type u_2
    ι : Type u_3
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    f : ι → α → M
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → Measurable (f i)
    ⊢ Measurable fun a => s.prod (fun c => f c) a
  -/
  exact Finset.prod_induction _ _ (fun _ _ => Measurable.mul) (@measurable_one M _ _ _ _) hf
  /-
    🎉 no goals
  -/


/-- Compositional version of `Finset.measurable_prod` for use by `fun_prop`. -/
@[to_additive (attr := measurability, fun_prop)
"Compositional version of `Finset.measurable_sum` for use by `fun_prop`."]
lemma Finset.measurable_prod' {f : ι → α → β → M} {g : α → β} {s : Finset ι}
    (hf : ∀ i, Measurable ↿(f i)) (hg : Measurable g) :
                                                    /-
                                                      M : Type u_2
                                                      ι : Type u_3
                                                      α : Type u_4
                                                      β : Type u_5
                                                      inst✝² : CommMonoid M
                                                      inst✝¹ : MeasurableSpace M
                                                      inst✝ : MeasurableMul₂ M
                                                      m : MeasurableSpace α
                                                      mβ : MeasurableSpace β
                                                      f : ι → α → β → M
                                                      g : α → β
                                                      s : Finset ι
                                                      hf : ∀ (i : ι), Measurable (Function.HasUncurry.uncurry (f i))
                                                      hg : Measurable g
                                                      ⊢ Measurable fun a => s.prod (fun i => f i a) (g a)
                                                    -/
    Measurable fun a ↦ (∏ i ∈ s, f i a) (g a) := by simp; fun_prop
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive (attr := measurability)]
theorem Finset.aemeasurable_prod' (s : Finset ι) (hf : ∀ i ∈ s, AEMeasurable (f i) μ) :
    AEMeasurable (∏ i ∈ s, f i) μ :=
  Multiset.aemeasurable_prod' _ fun _g hg =>
    let ⟨_i, hi, hg⟩ := Multiset.mem_map.1 hg
    hg ▸ hf _ hi


@[to_additive (attr := measurability)]
theorem Finset.aemeasurable_prod (s : Finset ι) (hf : ∀ i ∈ s, AEMeasurable (f i) μ) :
    AEMeasurable (fun a => ∏ i ∈ s, f i a) μ := by
  /-
    M : Type u_2
    ι : Type u_3
    α : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : MeasurableSpace M
    inst✝ : MeasurableMul₂ M
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : ι → α → M
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun a => s.prod fun i => f i a) μ
  -/
  simpa only [← Finset.prod_apply] using s.aemeasurable_prod' hf
  /-
    🎉 no goals
  -/


@[to_additive] -- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableMul [DiscreteMeasurableSpace α] :
    MeasurableMul α where
  measurable_const_mul _ := .of_discrete
  measurable_mul_const _ := .of_discrete


@[to_additive DiscreteMeasurableSpace.toMeasurableAdd₂] -- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableMul₂
    [DiscreteMeasurableSpace (α × α)] : MeasurableMul₂ α := ⟨.of_discrete⟩


@[to_additive] -- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableInv [DiscreteMeasurableSpace α] :
    MeasurableInv α := ⟨.of_discrete⟩


@[to_additive] -- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableDiv [DiscreteMeasurableSpace α] :
    MeasurableDiv α where
  measurable_const_div _ := .of_discrete
  measurable_div_const _ := .of_discrete


@[to_additive DiscreteMeasurableSpace.toMeasurableSub₂] -- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableDiv₂
    [DiscreteMeasurableSpace (α × α)] : MeasurableDiv₂ α := ⟨.of_discrete⟩

