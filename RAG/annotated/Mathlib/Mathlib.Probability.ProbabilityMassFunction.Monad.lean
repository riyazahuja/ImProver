/-- The pure `PMF` is the `PMF` where all the mass lies in one point.
  The value of `pure a` is `1` at `a` and `0` elsewhere. -/
def pure (a : α) : PMF α :=
  ⟨fun a' => if a' = a then 1 else 0, hasSum_ite_eq _ _⟩


@[simp]
theorem pure_apply : pure a a' = if a' = a then 1 else 0 := rfl


@[simp]
theorem support_pure : (pure a).support = {a} :=
                       /-
                         α : Type u_1
                         a a' : α
                         ⊢ Iff (Membership.mem (PMF.pure a).support a') (Membership.mem (Singleton.sing …
                       -/
  Set.ext fun a' => by simp [mem_support_iff]
                       /-
                         🎉 no goals
                       -/


                                                                    /-
                                                                      α : Type u_1
                                                                      a a' : α
                                                                      ⊢ Iff (Membership.mem (PMF.pure a).support a') (Eq a' a)
                                                                    -/
theorem mem_support_pure_iff : a' ∈ (pure a).support ↔ a' = a := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem pure_apply_self : pure a a = 1 :=
  if_pos rfl


theorem pure_apply_of_ne (h : a' ≠ a) : pure a a' = 0 :=
  if_neg h


instance [Inhabited α] : Inhabited (PMF α) :=
  ⟨pure default⟩


@[simp]
theorem toOuterMeasure_pure_apply : (pure a).toOuterMeasure s = if a ∈ s then 1 else 0 := by
  /-
    α : Type u_1
    a : α
    s : Set α
    ⊢ Eq ((PMF.pure a).toOuterMeasure s) (ite (Membership.mem s a) 1 0)
  -/
  refine (toOuterMeasure_apply (pure a) s).trans ?_
  /-
    α : Type u_1
    a : α
    s : Set α
    ⊢ Eq (tsum fun x => s.indicator (⇑(PMF.pure a)) x) (ite (Membership.mem s a) 1 …
  -/
  split_ifs with ha
    /-
      case pos
      α : Type u_1
      a : α
      s : Set α
      ha : Membership.mem s a
      ⊢ Eq (tsum fun x => s.indicator (⇑(PMF.pure a)) x) 1
    -/
  · refine (tsum_congr fun b => ?_).trans (tsum_ite_eq a 1)
    exact ite_eq_left_iff.2 fun hb =>
      symm (ite_eq_right_iff.2 fun h => (hb <| h.symm ▸ ha).elim)
    /-
      case neg
      α : Type u_1
      a : α
      s : Set α
      ha : Not (Membership.mem s a)
      ⊢ Eq (tsum fun x => s.indicator (⇑(PMF.pure a)) x) 0
    -/
  · refine (tsum_congr fun b => ?_).trans tsum_zero
    exact ite_eq_right_iff.2 fun hb =>
      ite_eq_right_iff.2 fun h => (ha <| h ▸ hb).elim


/-- The measure of a set under `pure a` is `1` for sets containing `a` and `0` otherwise. -/
@[simp]
theorem toMeasure_pure_apply (hs : MeasurableSet s) :
    (pure a).toMeasure s = if a ∈ s then 1 else 0 :=
  (toMeasure_apply_eq_toOuterMeasure_apply (pure a) s hs).trans (toOuterMeasure_pure_apply a s)


theorem toMeasure_pure : (pure a).toMeasure = Measure.dirac a :=
                             /-
                               α : Type u_1
                               a : α
                               inst✝ : MeasurableSpace α
                               s : Set α
                               hs : MeasurableSet s
                               ⊢ Eq ((PMF.pure a).toMeasure s) ((MeasureTheory.Measure.dirac a) s)
                             -/
  Measure.ext fun s hs => by rw [toMeasure_pure_apply a s hs, Measure.dirac_apply' a hs]; rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem toPMF_dirac [Countable α] [h : MeasurableSingletonClass α] :
    (Measure.dirac a).toPMF = pure a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : MeasurableSpace α
    inst✝ : Countable α
    h : MeasurableSingletonClass α
    ⊢ Eq (MeasureTheory.Measure.dirac a).toPMF (PMF.pure a)
  -/
  rw [toPMF_eq_iff_toMeasure_eq, toMeasure_pure]
  /-
    🎉 no goals
  -/


/-- The monadic bind operation for `PMF`. -/
def bind (p : PMF α) (f : α → PMF β) : PMF β :=
  ⟨fun b => ∑' a, p a * f a b,
    ENNReal.summable.hasSum_iff.2
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       γ : Type u_3
                                       p : PMF α
                                       f : α → PMF β
                                       ⊢ Eq (tsum fun b => tsum fun a => HMul.hMul (p b) ((f b) a)) 1
                                     -/
      (ENNReal.tsum_comm.trans <| by simp only [ENNReal.tsum_mul_left, tsum_coe, mul_one])⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem bind_apply (b : β) : p.bind f b = ∑' a, p a * f a b := rfl


@[simp]
theorem support_bind : (p.bind f).support = ⋃ a ∈ p.support, (f a).support :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        p : PMF α
                        f : α → PMF β
                        b : β
                        ⊢ Iff (Membership.mem (p.bind f).support b) (Membership.mem (Set.iUnion fun a  …
                      -/
  Set.ext fun b => by simp [mem_support_iff, ENNReal.tsum_eq_zero, not_or]
                      /-
                        🎉 no goals
                      -/


theorem mem_support_bind_iff (b : β) :
    b ∈ (p.bind f).support ↔ ∃ a ∈ p.support, b ∈ (f a).support := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : α → PMF β
    b : β
    ⊢ Iff (Membership.mem (p.bind f).support b) (Exists fun a => And (Membership.m …
  -/
  simp only [support_bind, Set.mem_iUnion, Set.mem_setOf_eq, exists_prop]
  /-
    🎉 no goals
  -/


@[simp]
theorem pure_bind (a : α) (f : α → PMF β) : (pure a).bind f = f a := by
  have : ∀ b a', ite (a' = a) (f a' b) 0 = ite (a' = a) (f a b) 0 := fun b a' => by
    split_ifs with h <;> simp [h]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : α → PMF β
    this : ∀ (b : β) (a' : α), Eq (ite (Eq a' a) ((f a') b) 0) (ite (Eq a' a) ((f  …
    ⊢ Eq ((PMF.pure a).bind f) (f a)
  -/
  ext b
  /-
    case h
    α : Type u_1
    β : Type u_2
    a : α
    f : α → PMF β
    this : ∀ (b : β) (a' : α), Eq (ite (Eq a' a) ((f a') b) 0) (ite (Eq a' a) ((f  …
    b : β
    ⊢ Eq (((PMF.pure a).bind f) b) ((f a) b)
  -/
  simp [this]
  /-
    🎉 no goals
  -/


@[simp]
theorem bind_pure : p.bind pure = p :=
  PMF.ext fun x => (bind_apply _ _ _).trans (_root_.trans
                                     /-
                                       α : Type u_1
                                       p : PMF α
                                       x y : α
                                       hy : Ne y x
                                       ⊢ Eq (HMul.hMul (p y) ((PMF.pure y) x)) 0
                                     -/
    (tsum_eq_single x fun y hy => by rw [pure_apply_of_ne _ _ hy.symm, mul_zero]) <|
                                     /-
                                       🎉 no goals
                                     -/
       /-
         α : Type u_1
         p : PMF α
         x : α
         ⊢ Eq (HMul.hMul (p x) ((PMF.pure x) x)) (p x)
       -/
    by rw [pure_apply_self, mul_one])
       /-
         🎉 no goals
       -/


@[simp]
theorem bind_const (p : PMF α) (q : PMF β) : (p.bind fun _ => q) = q :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        p : PMF α
                        q : PMF β
                        x : β
                        ⊢ Eq ((p.bind fun x => q) x) (q x)
                      -/
  PMF.ext fun x => by rw [bind_apply, ENNReal.tsum_mul_right, tsum_coe, one_mul]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem bind_bind : (p.bind f).bind g = p.bind fun a => (f a).bind g :=
  PMF.ext fun b => by
    simpa only [ENNReal.coe_inj.symm, bind_apply, ENNReal.tsum_mul_left.symm,
      ENNReal.tsum_mul_right.symm, mul_assoc, mul_left_comm, mul_comm] using ENNReal.tsum_comm


theorem bind_comm (p : PMF α) (q : PMF β) (f : α → β → PMF γ) :
    (p.bind fun a => q.bind (f a)) = q.bind fun b => p.bind fun a => f a b :=
  PMF.ext fun b => by
    simpa only [ENNReal.coe_inj.symm, bind_apply, ENNReal.tsum_mul_left.symm,
      ENNReal.tsum_mul_right.symm, mul_assoc, mul_left_comm, mul_comm] using ENNReal.tsum_comm


@[simp]
theorem toOuterMeasure_bind_apply :
    (p.bind f).toOuterMeasure s = ∑' a, p a * (f a).toOuterMeasure s :=
  calc
    (p.bind f).toOuterMeasure s = ∑' b, if b ∈ s then ∑' a, p a * f a b else 0 := by
      /-
        α : Type u_1
        β : Type u_2
        p : PMF α
        f : α → PMF β
        s : Set β
        ⊢ Eq ((p.bind f).toOuterMeasure s) (tsum fun b => ite (Membership.mem s b) (ts …
      -/
      simp [toOuterMeasure_apply, Set.indicator_apply]
      /-
        🎉 no goals
      -/
                                                                               /-
                                                                                 α : Type u_1
                                                                                 β : Type u_2
                                                                                 p : PMF α
                                                                                 f : α → PMF β
                                                                                 s : Set β
                                                                                 b : β
                                                                                 ⊢ Eq (ite (Membership.mem s b) (tsum fun a => HMul.hMul (p a) ((f a) b)) 0) (t …
                                                                               -/
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
    _ = ∑' (b) (a), p a * if b ∈ s then f a b else 0 := tsum_congr fun b => by split_ifs <;> simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
    _ = ∑' (a) (b), p a * if b ∈ s then f a b else 0 :=
      (tsum_comm' ENNReal.summable (fun _ => ENNReal.summable) fun _ => ENNReal.summable)
    _ = ∑' a, p a * ∑' b, if b ∈ s then f a b else 0 := tsum_congr fun _ => ENNReal.tsum_mul_left
    _ = ∑' a, p a * ∑' b, if b ∈ s then f a b else 0 :=
                                                                                  /-
                                                                                    α : Type u_1
                                                                                    β : Type u_2
                                                                                    p : PMF α
                                                                                    f : α → PMF β
                                                                                    s : Set β
                                                                                    a : α
                                                                                    b : β
                                                                                    ⊢ Eq (ite (Membership.mem s b) ((f a) b) 0) (ite (Membership.mem s b) ((f a) b …
                                                                                  -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
      (tsum_congr fun a => (congr_arg fun x => p a * x) <| tsum_congr fun b => by split_ifs <;> rfl)
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
    _ = ∑' a, p a * (f a).toOuterMeasure s :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               p : PMF α
                               f : α → PMF β
                               s : Set β
                               a : α
                               ⊢ Eq (HMul.hMul (p a) (tsum fun b => ite (Membership.mem s b) ((f a) b) 0)) (H …
                             -/
      tsum_congr fun a => by simp only [toOuterMeasure_apply, Set.indicator_apply]
                             /-
                               🎉 no goals
                             -/


/-- The measure of a set under `p.bind f` is the sum over `a : α`
  of the probability of `a` under `p` times the measure of the set under `f a`. -/
@[simp]
theorem toMeasure_bind_apply [MeasurableSpace β] (hs : MeasurableSet s) :
    (p.bind f).toMeasure s = ∑' a, p a * (f a).toMeasure s :=
  (toMeasure_apply_eq_toOuterMeasure_apply (p.bind f) s hs).trans
    ((toOuterMeasure_bind_apply p f s).trans
      (tsum_congr fun a =>
        congr_arg (fun x => p a * x) (toMeasure_apply_eq_toOuterMeasure_apply (f a) s hs).symm))


instance : Monad PMF where
  pure a := pure a
  bind pa pb := pa.bind pb


/-- Generalized version of `bind` allowing `f` to only be defined on the support of `p`.
  `p.bind f` is equivalent to `p.bindOnSupport (fun a _ ↦ f a)`, see `bindOnSupport_eq_bind`. -/
def bindOnSupport (p : PMF α) (f : ∀ a ∈ p.support, PMF β) : PMF β :=
  ⟨fun b => ∑' a, p a * if h : p a = 0 then 0 else f a h b, ENNReal.summable.hasSum_iff.2 (by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      ⊢ Eq (tsum fun b => tsum fun a => HMul.hMul (p a) (dite (Eq (p a) 0) (fun h => …
    -/
    refine ENNReal.tsum_comm.trans (_root_.trans (tsum_congr fun a => ?_) p.tsum_coe)
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      a : α
      ⊢ Eq (tsum fun a_1 => HMul.hMul (p a) (dite (Eq (p a) 0) (fun h => 0) fun h => …
    -/
    simp_rw [ENNReal.tsum_mul_left]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      a : α
      ⊢ Eq (HMul.hMul (p a) (tsum fun i => dite (Eq (p a) 0) (fun h => 0) fun h => ( …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        p : PMF α
        f : (a : α) → Membership.mem p.support a → PMF β
        a : α
        h : Eq (p a) 0
        ⊢ Eq (HMul.hMul (p a) (tsum fun i => 0)) (p a)
      -/
    · simp only [h, zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        p : PMF α
        f : (a : α) → Membership.mem p.support a → PMF β
        a : α
        h : Not (Eq (p a) 0)
        ⊢ Eq (HMul.hMul (p a) (tsum fun i => (f a h) i)) (p a)
      -/
    · rw [(f a h).tsum_coe, mul_one])⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem bindOnSupport_apply (b : β) :
    p.bindOnSupport f b = ∑' a, p a * if h : p a = 0 then 0 else f a h b := rfl


@[simp]
theorem support_bindOnSupport :
    (p.bindOnSupport f).support = ⋃ (a : α) (h : a ∈ p.support), (f a h).support := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    ⊢ Eq (p.bindOnSupport f).support (Set.iUnion fun a => Set.iUnion fun h => (f a …
  -/
  refine Set.ext fun b => ?_
  simp only [ENNReal.tsum_eq_zero, not_or, mem_support_iff, bindOnSupport_apply, Ne, not_forall,
    mul_eq_zero, Set.mem_iUnion]
  exact
    ⟨fun hb =>
      let ⟨a, ⟨ha, ha'⟩⟩ := hb
      ⟨a, ha, by simpa [ha] using ha'⟩,
      fun hb =>
      let ⟨a, ha, ha'⟩ := hb
      ⟨a, ⟨ha, by simpa [(mem_support_iff _ a).1 ha] using ha'⟩⟩⟩


theorem mem_support_bindOnSupport_iff (b : β) :
    b ∈ (p.bindOnSupport f).support ↔ ∃ (a : α) (h : a ∈ p.support), b ∈ (f a h).support := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    b : β
    ⊢ Iff (Membership.mem (p.bindOnSupport f).support b) (Exists fun a => Exists f …
  -/
  simp only [support_bindOnSupport, Set.mem_setOf_eq, Set.mem_iUnion]
  /-
    🎉 no goals
  -/


/-- `bindOnSupport` reduces to `bind` if `f` doesn't depend on the additional hypothesis. -/
@[simp]
theorem bindOnSupport_eq_bind (p : PMF α) (f : α → PMF β) :
    (p.bindOnSupport fun a _ => f a) = p.bind f := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : α → PMF β
    ⊢ Eq (p.bindOnSupport fun a x => f a) (p.bind f)
  -/
  ext b
  have : ∀ a, ite (p a = 0) 0 (p a * f a b) = p a * f a b :=
    fun a => ite_eq_right_iff.2 fun h => h.symm ▸ symm (zero_mul <| f a b)
  simp only [bindOnSupport_apply fun a _ => f a, p.bind_apply f, dite_eq_ite, mul_ite,
    mul_zero, this]


theorem bindOnSupport_eq_zero_iff (b : β) :
    p.bindOnSupport f b = 0 ↔ ∀ (a) (ha : p a ≠ 0), f a ha b = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    b : β
    ⊢ Iff (Eq ((p.bindOnSupport f) b) 0) (∀ (a : α) (ha : Ne (p a) 0), Eq ((f a ha …
  -/
  simp only [bindOnSupport_apply, ENNReal.tsum_eq_zero, mul_eq_zero, or_iff_not_imp_left]
  exact ⟨fun h a ha => Trans.trans (dif_neg ha).symm (h a ha),
    fun h a ha => Trans.trans (dif_neg ha) (h a ha)⟩


@[simp]
theorem pure_bindOnSupport (a : α) (f : ∀ (a' : α) (_ : a' ∈ (pure a).support), PMF β) :
    (pure a).bindOnSupport f = f a ((mem_support_pure_iff a a).mpr rfl) := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : (a' : α) → Membership.mem (PMF.pure a).support a' → PMF β
    ⊢ Eq ((PMF.pure a).bindOnSupport f) (f a ⋯)
  -/
  refine PMF.ext fun b => ?_
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : (a' : α) → Membership.mem (PMF.pure a).support a' → PMF β
    b : β
    ⊢ Eq (((PMF.pure a).bindOnSupport f) b) ((f a ⋯) b)
  -/
  simp only [bindOnSupport_apply, pure_apply]
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : (a' : α) → Membership.mem (PMF.pure a).support a' → PMF β
    b : β
    ⊢ Eq (tsum fun a_1 => HMul.hMul (ite (Eq a_1 a) 1 0) (dite (Eq (ite (Eq a_1 a) …
  -/
  refine _root_.trans (tsum_congr fun a' => ?_) (tsum_ite_eq a _)
  /-
    α : Type u_1
    β : Type u_2
    a : α
    f : (a' : α) → Membership.mem (PMF.pure a).support a' → PMF β
    b : β
    a' : α
    ⊢ Eq (HMul.hMul (ite (Eq a' a) 1 0) (dite (Eq (ite (Eq a' a) 1 0) 0) (fun h => …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases h : a' = a <;> simp [h]
                          /-
                            🎉 no goals
                          -/


theorem bindOnSupport_pure (p : PMF α) : (p.bindOnSupport fun a _ => pure a) = p := by
  /-
    α : Type u_1
    p : PMF α
    ⊢ Eq (p.bindOnSupport fun a x => PMF.pure a) p
  -/
  simp only [PMF.bind_pure, PMF.bindOnSupport_eq_bind]
  /-
    🎉 no goals
  -/


@[simp]
theorem bindOnSupport_bindOnSupport (p : PMF α) (f : ∀ a ∈ p.support, PMF β)
    (g : ∀ b ∈ (p.bindOnSupport f).support, PMF γ) :
    (p.bindOnSupport f).bindOnSupport g =
      p.bindOnSupport fun a ha =>
        (f a ha).bindOnSupport fun b hb =>
          g b ((mem_support_bindOnSupport_iff f b).mpr ⟨a, ha, hb⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    ⊢ Eq ((p.bindOnSupport f).bindOnSupport g) (p.bindOnSupport fun a ha => (f a h …
  -/
  refine PMF.ext fun a => ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    ⊢ Eq (((p.bindOnSupport f).bindOnSupport g) a) ((p.bindOnSupport fun a ha => ( …
  -/
  dsimp only [bindOnSupport_apply]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    ⊢ Eq (tsum fun a_1 => HMul.hMul (tsum fun a => HMul.hMul (p a) (dite (Eq (p a) …
  -/
  simp only [← tsum_dite_right, ENNReal.tsum_mul_left.symm, ENNReal.tsum_mul_right.symm]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    ⊢ Eq (tsum fun a_1 => tsum fun i => HMul.hMul (HMul.hMul (p i) (dite (Eq (p i) …
  -/
  simp only [ENNReal.tsum_eq_zero, dite_eq_left_iff]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    ⊢ Eq (tsum fun a_1 => tsum fun i => HMul.hMul (HMul.hMul (p i) (dite (Eq (p i) …
  -/
  refine ENNReal.tsum_comm.trans (tsum_congr fun a' => tsum_congr fun b => ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    a' : α
    b : β
    ⊢ Eq (HMul.hMul (HMul.hMul (p a') (dite (Eq (p a') 0) (fun h => 0) fun h => (f …
  -/
  split_ifs with h _ h_1 _ h_2
  /-
    case pos
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
    a : γ
    a' : α
    b : β
    h : Eq (p a') 0
    _ : ∀ (i : α), Eq (HMul.hMul (p i) (dite (Eq (p i) 0) (fun h => 0) fun h => (f …
    ⊢ Eq (HMul.hMul (HMul.hMul (p a') 0) 0) (HMul.hMul (p a') 0)
  -/
  any_goals ring1
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
      a : γ
      a' : α
      b : β
      h : Not (Eq (p a') 0)
      h_1 : ∀ (i : α), Eq (HMul.hMul (p i) (dite (Eq (p i) 0) (fun h => 0) fun h =>  …
      _ : Not (Eq ((f a' h) b) 0)
      ⊢ Eq (HMul.hMul (HMul.hMul (p a') ((f a' h) b)) 0) (HMul.hMul (p a') (HMul.hMu …
    -/
  · have := h_1 a'
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
      a : γ
      a' : α
      b : β
      h : Not (Eq (p a') 0)
      h_1 : ∀ (i : α), Eq (HMul.hMul (p i) (dite (Eq (p i) 0) (fun h => 0) fun h =>  …
      _ : Not (Eq ((f a' h) b) 0)
      this : Eq (HMul.hMul (p a') (dite (Eq (p a') 0) (fun h => 0) fun h => (f a' h) …
      ⊢ Eq (HMul.hMul (HMul.hMul (p a') ((f a' h) b)) 0) (HMul.hMul (p a') (HMul.hMu …
    -/
    simp? [h] at this says simp only [h, ↓reduceDIte, mul_eq_zero, false_or] at this
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
      a : γ
      a' : α
      b : β
      h : Not (Eq (p a') 0)
      h_1 : ∀ (i : α), Eq (HMul.hMul (p i) (dite (Eq (p i) 0) (fun h => 0) fun h =>  …
      _ : Not (Eq ((f a' h) b) 0)
      this : Eq ((f a' ⋯) b) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (p a') ((f a' h) b)) 0) (HMul.hMul (p a') (HMul.hMu …
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      p : PMF α
      f : (a : α) → Membership.mem p.support a → PMF β
      g : (b : β) → Membership.mem (p.bindOnSupport f).support b → PMF γ
      a : γ
      a' : α
      b : β
      h : Not (Eq (p a') 0)
      h_1 : Not (∀ (i : α), Eq (HMul.hMul (p i) (dite (Eq (p i) 0) (fun h => 0) fun  …
      h_2 : Eq ((f a' h) b) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (p a') ((f a' h) b)) ((g b ⋯) a)) (HMul.hMul (p a') …
    -/
  · simp [h_2]
    /-
      🎉 no goals
    -/


theorem bindOnSupport_comm (p : PMF α) (q : PMF β) (f : ∀ a ∈ p.support, ∀ b ∈ q.support, PMF γ) :
    (p.bindOnSupport fun a ha => q.bindOnSupport (f a ha)) =
      q.bindOnSupport fun b hb => p.bindOnSupport fun a ha => f a ha b hb := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    q : PMF β
    f : (a : α) → Membership.mem p.support a → (b : β) → Membership.mem q.support  …
    ⊢ Eq (p.bindOnSupport fun a ha => q.bindOnSupport (f a ha)) (q.bindOnSupport f …
  -/
  apply PMF.ext; rintro c
  simp only [ENNReal.coe_inj.symm, bindOnSupport_apply, ← tsum_dite_right,
    ENNReal.tsum_mul_left.symm, ENNReal.tsum_mul_right.symm]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    q : PMF β
    f : (a : α) → Membership.mem p.support a → (b : β) → Membership.mem q.support  …
    c : γ
    ⊢ Eq (tsum fun a => tsum fun i => HMul.hMul (p a) (dite (Eq (p a) 0) (fun h => …
  -/
  refine _root_.trans ENNReal.tsum_comm (tsum_congr fun b => tsum_congr fun a => ?_)
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    p : PMF α
    q : PMF β
    f : (a : α) → Membership.mem p.support a → (b : β) → Membership.mem q.support  …
    c : γ
    b : β
    a : α
    ⊢ Eq (HMul.hMul (p a) (dite (Eq (p a) 0) (fun h => 0) fun h => HMul.hMul (q b) …
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
  split_ifs with h1 h2 h2 <;> ring
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem toOuterMeasure_bindOnSupport_apply :
    (p.bindOnSupport f).toOuterMeasure s =
      ∑' a, p a * if h : p a = 0 then 0 else (f a h).toOuterMeasure s := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    s : Set β
    ⊢ Eq ((p.bindOnSupport f).toOuterMeasure s) (tsum fun a => HMul.hMul (p a) (di …
  -/
  simp only [toOuterMeasure_apply, Set.indicator_apply, bindOnSupport_apply]
  calc
    (∑' b, ite (b ∈ s) (∑' a, p a * dite (p a = 0) (fun h => 0) fun h => f a h b) 0) =
        ∑' (b) (a), ite (b ∈ s) (p a * dite (p a = 0) (fun h => 0) fun h => f a h b) 0 :=
      tsum_congr fun b => by split_ifs with hbs <;> simp only [eq_self_iff_true, tsum_zero]
    _ = ∑' (a) (b), ite (b ∈ s) (p a * dite (p a = 0) (fun h => 0) fun h => f a h b) 0 :=
      ENNReal.tsum_comm
    _ = ∑' a, p a * ∑' b, ite (b ∈ s) (dite (p a = 0) (fun h => 0) fun h => f a h b) 0 :=
      (tsum_congr fun a => by simp only [← ENNReal.tsum_mul_left, mul_ite, mul_zero])
    _ = ∑' a, p a * dite (p a = 0) (fun h => 0) fun h => ∑' b, ite (b ∈ s) (f a h b) 0 :=
      tsum_congr fun a => by split_ifs with ha <;> simp only [ite_self, tsum_zero, eq_self_iff_true]


/-- The measure of a set under `p.bindOnSupport f` is the sum over `a : α`
  of the probability of `a` under `p` times the measure of the set under `f a _`.
  The additional if statement is needed since `f` is only a partial function. -/
@[simp]
theorem toMeasure_bindOnSupport_apply [MeasurableSpace β] (hs : MeasurableSet s) :
    (p.bindOnSupport f).toMeasure s =
      ∑' a, p a * if h : p a = 0 then 0 else (f a h).toMeasure s := by
  /-
    α : Type u_1
    β : Type u_2
    p : PMF α
    f : (a : α) → Membership.mem p.support a → PMF β
    s : Set β
    inst✝ : MeasurableSpace β
    hs : MeasurableSet s
    ⊢ Eq ((p.bindOnSupport f).toMeasure s) (tsum fun a => HMul.hMul (p a) (dite (E …
  -/
  simp only [toMeasure_apply_eq_toOuterMeasure_apply _ _ hs, toOuterMeasure_bindOnSupport_apply]
  /-
    🎉 no goals
  -/


