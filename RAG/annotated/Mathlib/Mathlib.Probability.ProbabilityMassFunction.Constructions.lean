/-- The functorial action of a function on a `PMF`. -/
def map (f : α → β) (p : PMF α) : PMF β :=
  bind p (pure ∘ f)


theorem monad_map_eq_map {α β : Type u} (f : α → β) (p : PMF α) : f <$> p = p.map f := rfl


@[simp]
                                                                         /-
                                                                           α : Type u_1
                                                                           β : Type u_2
                                                                           f : α → β
                                                                           p : PMF α
                                                                           b : β
                                                                           ⊢ Eq ((PMF.map f p) b) (tsum fun a => ite (Eq b (f a)) (p a) 0)
                                                                         -/
theorem map_apply : (map f p) b = ∑' a, if b = f a then p a else 0 := by simp [map]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem support_map : (map f p).support = f '' p.support :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        f : α → β
                        p : PMF α
                        b : β
                        ⊢ Iff (Membership.mem (PMF.map f p).support b) (Membership.mem (Set.image f p. …
                      -/
  Set.ext fun b => by simp [map, @eq_comm β b]
                      /-
                        🎉 no goals
                      -/


                                                                                     /-
                                                                                       α : Type u_1
                                                                                       β : Type u_2
                                                                                       f : α → β
                                                                                       p : PMF α
                                                                                       b : β
                                                                                       ⊢ Iff (Membership.mem (PMF.map f p).support b) (Exists fun a => And (Membershi …
                                                                                     -/
theorem mem_support_map_iff : b ∈ (map f p).support ↔ ∃ a ∈ p.support, f a = b := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem bind_pure_comp : bind p (pure ∘ f) = map f p := rfl


theorem map_id : map id p = p :=
  bind_pure _


                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       γ : Type u_3
                                                                       f : α → β
                                                                       p : PMF α
                                                                       g : β → γ
                                                                       ⊢ Eq (PMF.map g (PMF.map f p)) (PMF.map (Function.comp g f) p)
                                                                     -/
theorem map_comp (g : β → γ) : (p.map f).map g = p.map (g ∘ f) := by simp [map, Function.comp_def]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem pure_map (a : α) : (pure a).map f = pure (f a) :=
  pure_bind _ _


theorem map_bind (q : α → PMF β) (f : β → γ) : (p.bind q).map f = p.bind fun a => (q a).map f :=
  bind_bind _ _ _


@[simp]
theorem bind_map (p : PMF α) (f : α → β) (q : β → PMF γ) : (p.map f).bind q = p.bind (q ∘ f) :=
  (bind_bind _ _ _).trans (congr_arg _ (funext fun _ => pure_bind _ _))


@[simp]
theorem map_const : p.map (Function.const α b) = pure b := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    b : β
    ⊢ Eq (PMF.map (Function.const α b) p) (PMF.pure b)
  -/
  simp only [map, Function.comp_def, bind_const, Function.const]
  /-
    🎉 no goals
  -/


@[simp]
theorem toOuterMeasure_map_apply : (p.map f).toOuterMeasure s = p.toOuterMeasure (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    p : PMF α
    s : Set β
    ⊢ Eq ((PMF.map f p).toOuterMeasure s) (p.toOuterMeasure (Set.preimage f s))
  -/
  simp [map, Set.indicator, toOuterMeasure_apply p (f ⁻¹' s)]
  /-
    🎉 no goals
  -/


@[simp]
theorem toMeasure_map_apply (hf : Measurable f)
    (hs : MeasurableSet s) : (p.map f).toMeasure s = p.toMeasure (f ⁻¹' s) := by
  rw [toMeasure_apply_eq_toOuterMeasure_apply _ s hs,
    toMeasure_apply_eq_toOuterMeasure_apply _ (f ⁻¹' s) (measurableSet_preimage hf hs)]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    p : PMF α
    s : Set β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    hf : Measurable f
    hs : MeasurableSet s
    ⊢ Eq ((PMF.map f p).toOuterMeasure s) (p.toOuterMeasure (Set.preimage f s))
  -/
  exact toOuterMeasure_map_apply f p s
  /-
    🎉 no goals
  -/


@[simp]
lemma toMeasure_map (p : PMF α) (hf : Measurable f) : p.toMeasure.map f = (p.map f).toMeasure := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    p : PMF α
    hf : Measurable f
    ⊢ Eq (MeasureTheory.Measure.map f p.toMeasure) (PMF.map f p).toMeasure
  -/
  ext s hs : 1; rw [PMF.toMeasure_map_apply _ _ _ hf hs, Measure.map_apply hf hs]
                /-
                  🎉 no goals
                -/


/-- The monadic sequencing operation for `PMF`. -/
def seq (q : PMF (α → β)) (p : PMF α) : PMF β :=
  q.bind fun m => p.bind fun a => pure (m a)


theorem monad_seq_eq_seq {α β : Type u} (q : PMF (α → β)) (p : PMF α) : q <*> p = q.seq p := rfl


@[simp]
theorem seq_apply : (seq q p) b = ∑' (f : α → β) (a : α), if b = f a then q f * p a else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    q : PMF (α → β)
    p : PMF α
    b : β
    ⊢ Eq ((q.seq p) b) (tsum fun f => tsum fun a => ite (Eq b (f a)) (HMul.hMul (q …
  -/
  simp only [seq, mul_boole, bind_apply, pure_apply]
  /-
    α : Type u_1
    β : Type u_2
    q : PMF (α → β)
    p : PMF α
    b : β
    ⊢ Eq (tsum fun a => HMul.hMul (q a) (tsum fun a_1 => ite (Eq b (a a_1)) (p a_1 …
  -/
  refine tsum_congr fun f => ENNReal.tsum_mul_left.symm.trans (tsum_congr fun a => ?_)
  /-
    α : Type u_1
    β : Type u_2
    q : PMF (α → β)
    p : PMF α
    b : β
    f : α → β
    a : α
    ⊢ Eq (HMul.hMul (q f) (ite (Eq b (f a)) (p a) 0)) (ite (Eq b (f a)) (HMul.hMul …
  -/
  simpa only [mul_zero] using mul_ite (b = f a) (q f) (p a) 0
  /-
    🎉 no goals
  -/


@[simp]
theorem support_seq : (seq q p).support = ⋃ f ∈ q.support, f '' p.support :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        q : PMF (α → β)
                        p : PMF α
                        b : β
                        ⊢ Iff (Membership.mem (q.seq p).support b) (Membership.mem (Set.iUnion fun f = …
                      -/
  Set.ext fun b => by simp [-mem_support_iff, seq, @eq_comm β b]
                      /-
                        🎉 no goals
                      -/


                                                                                                /-
                                                                                                  α : Type u_1
                                                                                                  β : Type u_2
                                                                                                  q : PMF (α → β)
                                                                                                  p : PMF α
                                                                                                  b : β
                                                                                                  ⊢ Iff (Membership.mem (q.seq p).support b) (Exists fun f => And (Membership.me …
                                                                                                -/
theorem mem_support_seq_iff : b ∈ (seq q p).support ↔ ∃ f ∈ q.support, b ∈ f '' p.support := by simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


instance : LawfulFunctor PMF where
  map_const := rfl
  id_map := bind_pure
  comp_map _ _ _ := (map_comp _ _ _).symm


                              /-
                                α : Type u_1
                                β : Type u_2
                                γ : Type u_3
                                ⊢ ∀ {α β : Type u_4} (x : α) (y : PMF β), Eq (Functor.mapConst x y) (Functor.m …
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
instance : LawfulMonad PMF := LawfulMonad.mk'
                              /-
                                🎉 no goals
                              -/
  (bind_pure_comp := fun _ _ => rfl)
  (id_map := id_map)
  (pure_bind := pure_bind)
  (bind_assoc := bind_bind)


/--
This instance allows `do` notation for `PMF` to be used across universes, for instance as
```lean4
example {R : Type u} [Ring R] (x : PMF ℕ) : PMF R := do
  let ⟨n⟩ ← ULiftable.up x
  pure n
```
where `x` is in universe `0`, but the return value is in universe `u`.
-/
instance : ULiftable PMF.{u} PMF.{v} where
  congr e :=
    { toFun := map e, invFun := map e.symm
                              /-
                                α : Type u_1
                                β : Type u_2
                                γ : Type u_3
                                α✝ : Type u
                                β✝ : Type v
                                e : Equiv α✝ β✝
                                a : PMF α✝
                                ⊢ Eq (PMF.map (⇑e.symm) (PMF.map (⇑e) a)) a
                              -/
      left_inv := fun a => by simp [map_comp, map_id]
                              /-
                                🎉 no goals
                              -/
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 γ : Type u_3
                                 α✝ : Type u
                                 β✝ : Type v
                                 e : Equiv α✝ β✝
                                 a : PMF β✝
                                 ⊢ Eq (PMF.map (⇑e) (PMF.map (⇑e.symm) a)) a
                               -/
      right_inv := fun a => by simp [map_comp, map_id] }
                               /-
                                 🎉 no goals
                               -/


/-- Given a finset `s` and a function `f : α → ℝ≥0∞` with sum `1` on `s`,
  such that `f a = 0` for `a ∉ s`, we get a `PMF`. -/
def ofFinset (f : α → ℝ≥0∞) (s : Finset α) (h : ∑ a ∈ s, f a = 1)
    (h' : ∀ (a) (_ : a ∉ s), f a = 0) : PMF α :=
  ⟨f, h ▸ hasSum_sum_of_ne_finset_zero h'⟩


@[simp]
theorem ofFinset_apply (a : α) : ofFinset f s h h' a = f a := rfl


@[simp]
theorem support_ofFinset : (ofFinset f s h h').support = ↑s ∩ Function.support f :=
                      /-
                        α : Type u_1
                        f : α → ENNReal
                        s : Finset α
                        h : Eq (s.sum fun a => f a) 1
                        h' : ∀ (a : α), Not (Membership.mem s a) → Eq (f a) 0
                        a : α
                        ⊢ Iff (Membership.mem (PMF.ofFinset f s h h').support a) (Membership.mem (Inte …
                      -/
  Set.ext fun a => by simpa [mem_support_iff] using mt (h' a)
                      /-
                        🎉 no goals
                      -/


theorem mem_support_ofFinset_iff (a : α) : a ∈ (ofFinset f s h h').support ↔ a ∈ s ∧ f a ≠ 0 := by
  /-
    α : Type u_1
    f : α → ENNReal
    s : Finset α
    h : Eq (s.sum fun a => f a) 1
    h' : ∀ (a : α), Not (Membership.mem s a) → Eq (f a) 0
    a : α
    ⊢ Iff (Membership.mem (PMF.ofFinset f s h h').support a) (And (Membership.mem  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ofFinset_apply_of_not_mem {a : α} (ha : a ∉ s) : ofFinset f s h h' a = 0 :=
  h' a ha


@[simp]
theorem toOuterMeasure_ofFinset_apply :
    (ofFinset f s h h').toOuterMeasure t = ∑' x, t.indicator f x :=
  toOuterMeasure_apply (ofFinset f s h h') t


@[simp]
theorem toMeasure_ofFinset_apply [MeasurableSpace α] (ht : MeasurableSet t) :
    (ofFinset f s h h').toMeasure t = ∑' x, t.indicator f x :=
  (toMeasure_apply_eq_toOuterMeasure_apply _ t ht).trans (toOuterMeasure_ofFinset_apply h h' t)


/-- Given a finite type `α` and a function `f : α → ℝ≥0∞` with sum 1, we get a `PMF`. -/
def ofFintype [Fintype α] (f : α → ℝ≥0∞) (h : ∑ a, f a = 1) : PMF α :=
  ofFinset f Finset.univ h fun a ha => absurd (Finset.mem_univ a) ha


@[simp]
theorem ofFintype_apply (a : α) : ofFintype f h a = f a := rfl


@[simp]
theorem support_ofFintype : (ofFintype f h).support = Function.support f := rfl


theorem mem_support_ofFintype_iff (a : α) : a ∈ (ofFintype f h).support ↔ f a ≠ 0 := Iff.rfl


@[simp]
lemma map_ofFintype [Fintype β] (f : α → ℝ≥0∞) (h : ∑ a, f a = 1) (g : α → β) :
    (ofFintype f h).map g = ofFintype (fun b ↦ ∑ a with g a = b, f a)
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝¹ : Fintype α
            f✝ : α → ENNReal
            h✝ : Eq (Finset.univ.sum fun a => f✝ a) 1
            inst✝ : Fintype β
            f : α → ENNReal
            h : Eq (Finset.univ.sum fun a => f a) 1
            g : α → β
            ⊢ Eq (Finset.univ.sum fun a => (fun b => (Finset.filter (fun a => Eq (g a) b)  …
          -/
      (by simpa [Finset.sum_fiberwise_eq_sum_filter univ univ g f]) := by
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : α → ENNReal
    h : Eq (Finset.univ.sum fun a => f a) 1
    g : α → β
    ⊢ Eq (PMF.map g (PMF.ofFintype f h)) (PMF.ofFintype (fun b => (Finset.filter ( …
  -/
  ext b : 1
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : α → ENNReal
    h : Eq (Finset.univ.sum fun a => f a) 1
    g : α → β
    b : β
    ⊢ Eq ((PMF.map g (PMF.ofFintype f h)) b) ((PMF.ofFintype (fun b => (Finset.fil …
  -/
  simp only [sum_filter, eq_comm, map_apply, ofFintype_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : α → ENNReal
    h : Eq (Finset.univ.sum fun a => f a) 1
    g : α → β
    b : β
    ⊢ Eq (tsum fun a => ite (Eq b (g a)) (f a) 0) (Finset.univ.sum fun a => ite (E …
  -/
  exact tsum_eq_sum fun _ h ↦ (h <| mem_univ _).elim
  /-
    🎉 no goals
  -/


@[simp high]
theorem toOuterMeasure_ofFintype_apply : (ofFintype f h).toOuterMeasure s = ∑' x, s.indicator f x :=
  toOuterMeasure_apply (ofFintype f h) s


@[simp]
theorem toMeasure_ofFintype_apply [MeasurableSpace α] (hs : MeasurableSet s) :
    (ofFintype f h).toMeasure s = ∑' x, s.indicator f x :=
  (toMeasure_apply_eq_toOuterMeasure_apply _ s hs).trans (toOuterMeasure_ofFintype_apply h s)


/-- Given an `f` with non-zero and non-infinite sum, get a `PMF` by normalizing `f` by its `tsum`.
-/
def normalize (f : α → ℝ≥0∞) (hf0 : tsum f ≠ 0) (hf : tsum f ≠ ∞) : PMF α :=
  ⟨fun a => f a * (∑' x, f x)⁻¹,
    ENNReal.summable.hasSum_iff.2 (ENNReal.tsum_mul_right.trans (ENNReal.mul_inv_cancel hf0 hf))⟩


@[simp]
theorem normalize_apply (a : α) : (normalize f hf0 hf) a = f a * (∑' x, f x)⁻¹ := rfl


@[simp]
theorem support_normalize : (normalize f hf0 hf).support = Function.support f :=
                      /-
                        α : Type u_1
                        f : α → ENNReal
                        hf0 : Ne (tsum f) 0
                        hf : Ne (tsum f) Top.top
                        a : α
                        ⊢ Iff (Membership.mem (PMF.normalize f hf0 hf).support a) (Membership.mem (Fun …
                      -/
  Set.ext fun a => by simp [hf, mem_support_iff]
                      /-
                        🎉 no goals
                      -/


                                                                                             /-
                                                                                               α : Type u_1
                                                                                               f : α → ENNReal
                                                                                               hf0 : Ne (tsum f) 0
                                                                                               hf : Ne (tsum f) Top.top
                                                                                               a : α
                                                                                               ⊢ Iff (Membership.mem (PMF.normalize f hf0 hf).support a) (Ne (f a) 0)
                                                                                             -/
theorem mem_support_normalize_iff (a : α) : a ∈ (normalize f hf0 hf).support ↔ f a ≠ 0 := by simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


/-- Create new `PMF` by filtering on a set with non-zero measure and normalizing. -/
def filter (p : PMF α) (s : Set α) (h : ∃ a ∈ s, a ∈ p.support) : PMF α :=
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      γ : Type u_3
                                      p : PMF α
                                      s : Set α
                                      h : Exists fun a => And (Membership.mem s a) (Membership.mem p.support a)
                                      ⊢ Ne (tsum (s.indicator ⇑p)) 0
                                    -/
  PMF.normalize (s.indicator p) (by simpa using h) (p.tsum_coe_indicator_ne_top s)
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem filter_apply (a : α) :
    (p.filter s h) a = s.indicator p a * (∑' a', (s.indicator p) a')⁻¹ := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    h : Exists fun a => And (Membership.mem s a) (Membership.mem p.support a)
    a : α
    ⊢ Eq ((p.filter s h) a) (HMul.hMul (s.indicator (⇑p) a) (Inv.inv (tsum fun a'  …
  -/
  rw [filter, normalize_apply]
  /-
    🎉 no goals
  -/


theorem filter_apply_eq_zero_of_not_mem {a : α} (ha : a ∉ s) : (p.filter s h) a = 0 := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    h : Exists fun a => And (Membership.mem s a) (Membership.mem p.support a)
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Eq ((p.filter s h) a) 0
  -/
  rw [filter_apply, Set.indicator_apply_eq_zero.mpr fun ha' => absurd ha' ha, zero_mul]
  /-
    🎉 no goals
  -/


theorem mem_support_filter_iff {a : α} : a ∈ (p.filter s h).support ↔ a ∈ s ∧ a ∈ p.support :=
  (mem_support_normalize_iff _ _ _).trans Set.indicator_apply_ne_zero


@[simp]
theorem support_filter : (p.filter s h).support = s ∩ p.support :=
  Set.ext fun _ => mem_support_filter_iff _


theorem filter_apply_eq_zero_iff (a : α) : (p.filter s h) a = 0 ↔ a ∉ s ∨ a ∉ p.support := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    h : Exists fun a => And (Membership.mem s a) (Membership.mem p.support a)
    a : α
    ⊢ Iff (Eq ((p.filter s h) a) 0) (Or (Not (Membership.mem s a)) (Not (Membershi …
  -/
  rw [apply_eq_zero_iff, support_filter, Set.mem_inter_iff, not_and_or]
  /-
    🎉 no goals
  -/


theorem filter_apply_ne_zero_iff (a : α) : (p.filter s h) a ≠ 0 ↔ a ∈ s ∧ a ∈ p.support := by
  /-
    α : Type u_1
    p : PMF α
    s : Set α
    h : Exists fun a => And (Membership.mem s a) (Membership.mem p.support a)
    a : α
    ⊢ Iff (Ne ((p.filter s h) a) 0) (And (Membership.mem s a) (Membership.mem p.su …
  -/
  rw [Ne, filter_apply_eq_zero_iff, not_or, Classical.not_not, Classical.not_not]
  /-
    🎉 no goals
  -/


/-- A `PMF` which assigns probability `p` to `true` and `1 - p` to `false`. -/
def bernoulli (p : ℝ≥0∞) (h : p ≤ 1) : PMF Bool :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              γ : Type u_3
                                              p : ENNReal
                                              h : LE.le p 1
                                              ⊢ Eq (Finset.univ.sum fun a => (fun b => cond b p (HSub.hSub 1 p)) a) 1
                                            -/
  ofFintype (fun b => cond b p (1 - p)) (by simp [h])
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem bernoulli_apply : bernoulli p h b = cond b p (1 - p) := rfl


@[simp]
theorem support_bernoulli : (bernoulli p h).support = { b | cond b (p ≠ 0) (p ≠ 1) } := by
  /-
    p : ENNReal
    h : LE.le p 1
    ⊢ Eq (PMF.bernoulli p h).support (setOf fun b => cond b (Ne p 0) (Ne p 1))
  -/
  refine Set.ext fun b => ?_
  /-
    p : ENNReal
    h : LE.le p 1
    b : Bool
    ⊢ Iff (Membership.mem (PMF.bernoulli p h).support b) (Membership.mem (setOf fu …
  -/
  induction b
    /-
      case false
      p : ENNReal
      h : LE.le p 1
      ⊢ Iff (Membership.mem (PMF.bernoulli p h).support Bool.false) (Membership.mem  …
    -/
  · simp_rw [mem_support_iff, bernoulli_apply, Bool.cond_false, Ne, tsub_eq_zero_iff_le, not_le]
    /-
      case false
      p : ENNReal
      h : LE.le p 1
      ⊢ Iff (LT.lt p 1) (Membership.mem (setOf fun b => cond b (Not (Eq p 0)) (Not ( …
    -/
    exact ⟨ne_of_lt, lt_of_le_of_ne h⟩
    /-
      🎉 no goals
    -/
    /-
      case true
      p : ENNReal
      h : LE.le p 1
      ⊢ Iff (Membership.mem (PMF.bernoulli p h).support Bool.true) (Membership.mem ( …
    -/
  · simp only [mem_support_iff, bernoulli_apply, Bool.cond_true, Set.mem_setOf_eq]
    /-
      🎉 no goals
    -/


                                                                                               /-
                                                                                                 p : ENNReal
                                                                                                 h : LE.le p 1
                                                                                                 b : Bool
                                                                                                 ⊢ Iff (Membership.mem (PMF.bernoulli p h).support b) (cond b (Ne p 0) (Ne p 1))
                                                                                               -/
theorem mem_support_bernoulli_iff : b ∈ (bernoulli p h).support ↔ cond b (p ≠ 0) (p ≠ 1) := by simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


