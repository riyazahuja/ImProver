/-- Hyperreal numbers on the ultrafilter extending the cofinite filter -/
def Hyperreal : Type :=
  Germ (hyperfilter ℕ : Filter ℕ) ℝ deriving Inhabited


@[inherit_doc] notation "ℝ*" => Hyperreal


noncomputable instance : LinearOrderedField ℝ* :=
  inferInstanceAs (LinearOrderedField (Germ _ _))


/-- Natural embedding `ℝ → ℝ*`. -/
@[coe] def ofReal : ℝ → ℝ* := const


noncomputable instance : CoeTC ℝ ℝ* := ⟨ofReal⟩


@[simp, norm_cast]
theorem coe_eq_coe {x y : ℝ} : (x : ℝ*) = y ↔ x = y :=
  Germ.const_inj


theorem coe_ne_coe {x y : ℝ} : (x : ℝ*) ≠ y ↔ x ≠ y :=
  coe_eq_coe.not


@[simp, norm_cast]
theorem coe_eq_zero {x : ℝ} : (x : ℝ*) = 0 ↔ x = 0 :=
  coe_eq_coe


@[simp, norm_cast]
theorem coe_eq_one {x : ℝ} : (x : ℝ*) = 1 ↔ x = 1 :=
  coe_eq_coe


@[norm_cast]
theorem coe_ne_zero {x : ℝ} : (x : ℝ*) ≠ 0 ↔ x ≠ 0 :=
  coe_ne_coe


@[norm_cast]
theorem coe_ne_one {x : ℝ} : (x : ℝ*) ≠ 1 ↔ x ≠ 1 :=
  coe_ne_coe


@[simp, norm_cast]
theorem coe_one : ↑(1 : ℝ) = (1 : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_zero : ↑(0 : ℝ) = (0 : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_inv (x : ℝ) : ↑x⁻¹ = (x⁻¹ : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_neg (x : ℝ) : ↑(-x) = (-x : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_add (x y : ℝ) : ↑(x + y) = (x + y : ℝ*) :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp, norm_cast]
theorem coe_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((no_index (OfNat.ofNat n : ℝ)) : ℝ*) = OfNat.ofNat n :=
  rfl


@[simp, norm_cast]
theorem coe_mul (x y : ℝ) : ↑(x * y) = (x * y : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_div (x y : ℝ) : ↑(x / y) = (x / y : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_sub (x y : ℝ) : ↑(x - y) = (x - y : ℝ*) :=
  rfl


@[simp, norm_cast]
theorem coe_le_coe {x y : ℝ} : (x : ℝ*) ≤ y ↔ x ≤ y :=
  Germ.const_le_iff


@[simp, norm_cast]
theorem coe_lt_coe {x y : ℝ} : (x : ℝ*) < y ↔ x < y :=
  Germ.const_lt_iff


@[simp, norm_cast]
theorem coe_nonneg {x : ℝ} : 0 ≤ (x : ℝ*) ↔ 0 ≤ x :=
  coe_le_coe


@[simp, norm_cast]
theorem coe_pos {x : ℝ} : 0 < (x : ℝ*) ↔ 0 < x :=
  coe_lt_coe


@[simp, norm_cast]
theorem coe_abs (x : ℝ) : ((|x| : ℝ) : ℝ*) = |↑x| :=
  const_abs x


@[simp, norm_cast]
theorem coe_max (x y : ℝ) : ((max x y : ℝ) : ℝ*) = max ↑x ↑y :=
  Germ.const_max _ _


@[simp, norm_cast]
theorem coe_min (x y : ℝ) : ((min x y : ℝ) : ℝ*) = min ↑x ↑y :=
  Germ.const_min _ _


/-- Construct a hyperreal number from a sequence of real numbers. -/
def ofSeq (f : ℕ → ℝ) : ℝ* := (↑f : Germ (hyperfilter ℕ : Filter ℕ) ℝ)


theorem ofSeq_surjective : Function.Surjective ofSeq := Quot.exists_rep


theorem ofSeq_lt_ofSeq {f g : ℕ → ℝ} : ofSeq f < ofSeq g ↔ ∀ᶠ n in hyperfilter ℕ, f n < g n :=
  Germ.coe_lt


/-- A sample infinitesimal hyperreal -/
noncomputable def epsilon : ℝ* :=
  ofSeq fun n => n⁻¹


/-- A sample infinite hyperreal -/
noncomputable def omega : ℝ* := ofSeq Nat.cast


@[inherit_doc] scoped notation "ε" => Hyperreal.epsilon

@[inherit_doc] scoped notation "ω" => Hyperreal.omega


@[simp]
theorem inv_omega : ω⁻¹ = ε :=
  rfl


@[simp]
theorem inv_epsilon : ε⁻¹ = ω :=
  @inv_inv _ _ ω


theorem omega_pos : 0 < ω :=
  Germ.coe_pos.2 <| Nat.hyperfilter_le_atTop <| (eventually_gt_atTop 0).mono fun _ ↦
    Nat.cast_pos.2


theorem epsilon_pos : 0 < ε :=
  inv_pos_of_pos omega_pos


theorem epsilon_ne_zero : ε ≠ 0 :=
  epsilon_pos.ne'


theorem omega_ne_zero : ω ≠ 0 :=
  omega_pos.ne'


theorem epsilon_mul_omega : ε * ω = 1 :=
  @inv_mul_cancel₀ _ _ ω omega_ne_zero


theorem lt_of_tendsto_zero_of_pos {f : ℕ → ℝ} (hf : Tendsto f atTop (𝓝 0)) :
    ∀ {r : ℝ}, 0 < r → ofSeq f < (r : ℝ*) := fun hr ↦
  ofSeq_lt_ofSeq.2 <| (hf.eventually <| gt_mem_nhds hr).filter_mono Nat.hyperfilter_le_atTop


theorem neg_lt_of_tendsto_zero_of_pos {f : ℕ → ℝ} (hf : Tendsto f atTop (𝓝 0)) :
    ∀ {r : ℝ}, 0 < r → (-r : ℝ*) < ofSeq f := fun hr =>
  have hg := hf.neg
                       /-
                         f : Nat → Real
                         hf : Filter.Tendsto f Filter.atTop (nhds 0)
                         r✝ : Real
                         hr : LT.lt 0 r✝
                         hg : Filter.Tendsto (fun x => Neg.neg (f x)) Filter.atTop (nhds (-0))
                         ⊢ LT.lt (Neg.neg (Hyperreal.ofSeq f)) ↑r✝
                       -/
  neg_lt_of_neg_lt (by rw [neg_zero] at hg; exact lt_of_tendsto_zero_of_pos hg hr)
                                            /-
                                              🎉 no goals
                                            -/


theorem gt_of_tendsto_zero_of_neg {f : ℕ → ℝ} (hf : Tendsto f atTop (𝓝 0)) :
    ∀ {r : ℝ}, r < 0 → (r : ℝ*) < ofSeq f := fun {r} hr => by
  /-
    f : Nat → Real
    hf : Filter.Tendsto f Filter.atTop (nhds 0)
    r : Real
    hr : LT.lt r 0
    ⊢ LT.lt (↑r) (Hyperreal.ofSeq f)
  -/
  rw [← neg_neg r, coe_neg]; exact neg_lt_of_tendsto_zero_of_pos hf (neg_pos.mpr hr)
                             /-
                               🎉 no goals
                             -/


theorem epsilon_lt_pos (x : ℝ) : 0 < x → ε < x :=
  lt_of_tendsto_zero_of_pos tendsto_inverse_atTop_nhds_zero_nat


/-- Standard part predicate -/
def IsSt (x : ℝ*) (r : ℝ) :=
  ∀ δ : ℝ, 0 < δ → (r - δ : ℝ*) < x ∧ x < r + δ


open scoped Classical in
/-- Standard part function: like a "round" to ℝ instead of ℤ -/
noncomputable def st : ℝ* → ℝ := fun x => if h : ∃ r, IsSt x r then Classical.choose h else 0


/-- A hyperreal number is infinitesimal if its standard part is 0 -/
def Infinitesimal (x : ℝ*) :=
  IsSt x 0


/-- A hyperreal number is positive infinite if it is larger than all real numbers -/
def InfinitePos (x : ℝ*) :=
  ∀ r : ℝ, ↑r < x


/-- A hyperreal number is negative infinite if it is smaller than all real numbers -/
def InfiniteNeg (x : ℝ*) :=
  ∀ r : ℝ, x < r


/-- A hyperreal number is infinite if it is infinite positive or infinite negative -/
def Infinite (x : ℝ*) :=
  InfinitePos x ∨ InfiniteNeg x


theorem isSt_ofSeq_iff_tendsto {f : ℕ → ℝ} {r : ℝ} :
    IsSt (ofSeq f) r ↔ Tendsto f (hyperfilter ℕ) (𝓝 r) :=
  Iff.trans (forall₂_congr fun _ _ ↦ (ofSeq_lt_ofSeq.and ofSeq_lt_ofSeq).trans eventually_and.symm)
    (nhds_basis_Ioo_pos _).tendsto_right_iff.symm


theorem isSt_iff_tendsto {x : ℝ*} {r : ℝ} : IsSt x r ↔ x.Tendsto (𝓝 r) := by
  /-
    x : Hyperreal
    r : Real
    ⊢ Iff (x.IsSt r) (Filter.Germ.Tendsto x (nhds r))
  -/
  rcases ofSeq_surjective x with ⟨f, rfl⟩
  /-
    case intro
    r : Real
    f : Nat → Real
    ⊢ Iff ((Hyperreal.ofSeq f).IsSt r) (Filter.Germ.Tendsto (Hyperreal.ofSeq f) (n …
  -/
  exact isSt_ofSeq_iff_tendsto
  /-
    🎉 no goals
  -/


theorem isSt_of_tendsto {f : ℕ → ℝ} {r : ℝ} (hf : Tendsto f atTop (𝓝 r)) : IsSt (ofSeq f) r :=
  isSt_ofSeq_iff_tendsto.2 <| hf.mono_left Nat.hyperfilter_le_atTop

-- Porting note: moved up, renamed

protected theorem IsSt.lt {x y : ℝ*} {r s : ℝ} (hxr : IsSt x r) (hys : IsSt y s) (hrs : r < s) :
    x < y := by
  /-
    x y : Hyperreal
    r s : Real
    hxr : x.IsSt r
    hys : y.IsSt s
    hrs : LT.lt r s
    ⊢ LT.lt x y
  -/
  rcases ofSeq_surjective x with ⟨f, rfl⟩
  /-
    case intro
    y : Hyperreal
    r s : Real
    hys : y.IsSt s
    hrs : LT.lt r s
    f : Nat → Real
    hxr : (Hyperreal.ofSeq f).IsSt r
    ⊢ LT.lt (Hyperreal.ofSeq f) y
  -/
  rcases ofSeq_surjective y with ⟨g, rfl⟩
  /-
    case intro.intro
    r s : Real
    hrs : LT.lt r s
    f : Nat → Real
    hxr : (Hyperreal.ofSeq f).IsSt r
    g : Nat → Real
    hys : (Hyperreal.ofSeq g).IsSt s
    ⊢ LT.lt (Hyperreal.ofSeq f) (Hyperreal.ofSeq g)
  -/
  rw [isSt_ofSeq_iff_tendsto] at hxr hys
  /-
    case intro.intro
    r s : Real
    hrs : LT.lt r s
    f : Nat → Real
    hxr : Filter.Tendsto f (↑(Filter.hyperfilter Nat)) (nhds r)
    g : Nat → Real
    hys : Filter.Tendsto g (↑(Filter.hyperfilter Nat)) (nhds s)
    ⊢ LT.lt (Hyperreal.ofSeq f) (Hyperreal.ofSeq g)
  -/
  exact ofSeq_lt_ofSeq.2 <| hxr.eventually_lt hys hrs
  /-
    🎉 no goals
  -/


theorem IsSt.unique {x : ℝ*} {r s : ℝ} (hr : IsSt x r) (hs : IsSt x s) : r = s := by
  /-
    x : Hyperreal
    r s : Real
    hr : x.IsSt r
    hs : x.IsSt s
    ⊢ Eq r s
  -/
  rcases ofSeq_surjective x with ⟨f, rfl⟩
  /-
    case intro
    r s : Real
    f : Nat → Real
    hr : (Hyperreal.ofSeq f).IsSt r
    hs : (Hyperreal.ofSeq f).IsSt s
    ⊢ Eq r s
  -/
  rw [isSt_ofSeq_iff_tendsto] at hr hs
  /-
    case intro
    r s : Real
    f : Nat → Real
    hr : Filter.Tendsto f (↑(Filter.hyperfilter Nat)) (nhds r)
    hs : Filter.Tendsto f (↑(Filter.hyperfilter Nat)) (nhds s)
    ⊢ Eq r s
  -/
  exact tendsto_nhds_unique hr hs
  /-
    🎉 no goals
  -/


theorem IsSt.st_eq {x : ℝ*} {r : ℝ} (hxr : IsSt x r) : st x = r := by
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    ⊢ Eq x.st r
  -/
  have h : ∃ r, IsSt x r := ⟨r, hxr⟩
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    h : Exists fun r => x.IsSt r
    ⊢ Eq x.st r
  -/
  rw [st, dif_pos h]
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    h : Exists fun r => x.IsSt r
    ⊢ Eq (Classical.choose h) r
  -/
  exact (Classical.choose_spec h).unique hxr
  /-
    🎉 no goals
  -/


theorem IsSt.not_infinite {x : ℝ*} {r : ℝ} (h : IsSt x r) : ¬Infinite x := fun hi ↦
  hi.elim (fun hp ↦ lt_asymm (h 1 one_pos).2 (hp (r + 1))) fun hn ↦
    lt_asymm (h 1 one_pos).1 (hn (r - 1))


theorem not_infinite_of_exists_st {x : ℝ*} : (∃ r : ℝ, IsSt x r) → ¬Infinite x := fun ⟨_r, hr⟩ =>
  hr.not_infinite


theorem Infinite.st_eq {x : ℝ*} (hi : Infinite x) : st x = 0 :=
  dif_neg fun ⟨_r, hr⟩ ↦ hr.not_infinite hi


theorem isSt_sSup {x : ℝ*} (hni : ¬Infinite x) : IsSt x (sSup { y : ℝ | (y : ℝ*) < x }) :=
  let S : Set ℝ := { y : ℝ | (y : ℝ*) < x }
  let R : ℝ := sSup S
  let ⟨r₁, hr₁⟩ := not_forall.mp (not_or.mp hni).2
  let ⟨r₂, hr₂⟩ := not_forall.mp (not_or.mp hni).1
  have HR₁ : S.Nonempty :=
    ⟨r₁ - 1, lt_of_lt_of_le (coe_lt_coe.2 <| sub_one_lt _) (not_lt.mp hr₁)⟩
  have HR₂ : BddAbove S :=
    ⟨r₂, fun _y hy => le_of_lt (coe_lt_coe.1 (lt_of_lt_of_le hy (not_lt.mp hr₂)))⟩
  fun δ hδ =>
  ⟨lt_of_not_le fun c =>
      have hc : ∀ y ∈ S, y ≤ R - δ := fun _y hy =>
        coe_le_coe.1 <| le_of_lt <| lt_of_lt_of_le hy c
      not_lt_of_le (csSup_le HR₁ hc) <| sub_lt_self R hδ,
    lt_of_not_le fun c =>
      have hc : ↑(R + δ / 2) < x :=
        lt_of_lt_of_le (add_lt_add_left (coe_lt_coe.2 (half_lt_self hδ)) R) c
      not_lt_of_le (le_csSup HR₂ hc) <| (lt_add_iff_pos_right _).mpr <| half_pos hδ⟩


theorem exists_st_of_not_infinite {x : ℝ*} (hni : ¬Infinite x) : ∃ r : ℝ, IsSt x r :=
  ⟨sSup { y : ℝ | (y : ℝ*) < x }, isSt_sSup hni⟩


theorem st_eq_sSup {x : ℝ*} : st x = sSup { y : ℝ | (y : ℝ*) < x } := by
  /-
    x : Hyperreal
    ⊢ Eq x.st (SupSet.sSup (setOf fun y => LT.lt (↑y) x))
  -/
  rcases _root_.em (Infinite x) with (hx|hx)
    /-
      case inl
      x : Hyperreal
      hx : x.Infinite
      ⊢ Eq x.st (SupSet.sSup (setOf fun y => LT.lt (↑y) x))
    -/
  · rw [hx.st_eq]
    cases hx with
    | inl hx =>
      convert Real.sSup_univ.symm
      exact Set.eq_univ_of_forall hx
    | inr hx =>
      convert Real.sSup_empty.symm
      exact Set.eq_empty_of_forall_not_mem fun y hy ↦ hy.out.not_lt (hx _)
    /-
      case inr
      x : Hyperreal
      hx : Not x.Infinite
      ⊢ Eq x.st (SupSet.sSup (setOf fun y => LT.lt (↑y) x))
    -/
  · exact (isSt_sSup hx).st_eq
    /-
      🎉 no goals
    -/


theorem exists_st_iff_not_infinite {x : ℝ*} : (∃ r : ℝ, IsSt x r) ↔ ¬Infinite x :=
  ⟨not_infinite_of_exists_st, exists_st_of_not_infinite⟩


theorem infinite_iff_not_exists_st {x : ℝ*} : Infinite x ↔ ¬∃ r : ℝ, IsSt x r :=
  iff_not_comm.mp exists_st_iff_not_infinite


theorem IsSt.isSt_st {x : ℝ*} {r : ℝ} (hxr : IsSt x r) : IsSt x (st x) := by
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    ⊢ x.IsSt x.st
  -/
  rwa [hxr.st_eq]
  /-
    🎉 no goals
  -/


theorem isSt_st_of_exists_st {x : ℝ*} (hx : ∃ r : ℝ, IsSt x r) : IsSt x (st x) :=
  let ⟨_r, hr⟩ := hx; hr.isSt_st


theorem isSt_st' {x : ℝ*} (hx : ¬Infinite x) : IsSt x (st x) :=
  (isSt_sSup hx).isSt_st


theorem isSt_st {x : ℝ*} (hx : st x ≠ 0) : IsSt x (st x) :=
  isSt_st' <| mt Infinite.st_eq hx


theorem isSt_refl_real (r : ℝ) : IsSt r r := isSt_ofSeq_iff_tendsto.2 tendsto_const_nhds


theorem st_id_real (r : ℝ) : st r = r := (isSt_refl_real r).st_eq


theorem eq_of_isSt_real {r s : ℝ} : IsSt r s → r = s :=
  (isSt_refl_real r).unique


theorem isSt_real_iff_eq {r s : ℝ} : IsSt r s ↔ r = s :=
  ⟨eq_of_isSt_real, fun hrs => hrs ▸ isSt_refl_real r⟩


theorem isSt_symm_real {r s : ℝ} : IsSt r s ↔ IsSt s r := by
  /-
    r s : Real
    ⊢ Iff ((↑r).IsSt s) ((↑s).IsSt r)
  -/
  rw [isSt_real_iff_eq, isSt_real_iff_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem isSt_trans_real {r s t : ℝ} : IsSt r s → IsSt s t → IsSt r t := by
  /-
    r s t : Real
    ⊢ (↑r).IsSt s → (↑s).IsSt t → (↑r).IsSt t
  -/
  rw [isSt_real_iff_eq, isSt_real_iff_eq, isSt_real_iff_eq]; exact Eq.trans
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem isSt_inj_real {r₁ r₂ s : ℝ} (h1 : IsSt r₁ s) (h2 : IsSt r₂ s) : r₁ = r₂ :=
  Eq.trans (eq_of_isSt_real h1) (eq_of_isSt_real h2).symm


theorem isSt_iff_abs_sub_lt_delta {x : ℝ*} {r : ℝ} : IsSt x r ↔ ∀ δ : ℝ, 0 < δ → |x - ↑r| < δ := by
  /-
    x : Hyperreal
    r : Real
    ⊢ Iff (x.IsSt r) (∀ (δ : Real), LT.lt 0 δ → LT.lt (abs (HSub.hSub x ↑r)) ↑δ)
  -/
  simp only [abs_sub_lt_iff, sub_lt_iff_lt_add, IsSt, and_comm, add_comm]
  /-
    🎉 no goals
  -/


theorem IsSt.map {x : ℝ*} {r : ℝ} (hxr : IsSt x r) {f : ℝ → ℝ} (hf : ContinuousAt f r) :
    IsSt (x.map f) (f r) := by
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    f : Real → Real
    hf : ContinuousAt f r
    ⊢ Hyperreal.IsSt (Filter.Germ.map f x) (f r)
  -/
  rcases ofSeq_surjective x with ⟨g, rfl⟩
  /-
    case intro
    r : Real
    f : Real → Real
    hf : ContinuousAt f r
    g : Nat → Real
    hxr : (Hyperreal.ofSeq g).IsSt r
    ⊢ Hyperreal.IsSt (Filter.Germ.map f (Hyperreal.ofSeq g)) (f r)
  -/
  exact isSt_ofSeq_iff_tendsto.2 <| hf.tendsto.comp (isSt_ofSeq_iff_tendsto.1 hxr)
  /-
    🎉 no goals
  -/


theorem IsSt.map₂ {x y : ℝ*} {r s : ℝ} (hxr : IsSt x r) (hys : IsSt y s) {f : ℝ → ℝ → ℝ}
    (hf : ContinuousAt (Function.uncurry f) (r, s)) : IsSt (x.map₂ f y) (f r s) := by
  /-
    x y : Hyperreal
    r s : Real
    hxr : x.IsSt r
    hys : y.IsSt s
    f : Real → Real → Real
    hf : ContinuousAt (Function.uncurry f) { fst := r, snd := s }
    ⊢ Hyperreal.IsSt (Filter.Germ.map₂ f x y) (f r s)
  -/
  rcases ofSeq_surjective x with ⟨x, rfl⟩
  /-
    case intro
    y : Hyperreal
    r s : Real
    hys : y.IsSt s
    f : Real → Real → Real
    hf : ContinuousAt (Function.uncurry f) { fst := r, snd := s }
    x : Nat → Real
    hxr : (Hyperreal.ofSeq x).IsSt r
    ⊢ Hyperreal.IsSt (Filter.Germ.map₂ f (Hyperreal.ofSeq x) y) (f r s)
  -/
  rcases ofSeq_surjective y with ⟨y, rfl⟩
  /-
    case intro.intro
    r s : Real
    f : Real → Real → Real
    hf : ContinuousAt (Function.uncurry f) { fst := r, snd := s }
    x : Nat → Real
    hxr : (Hyperreal.ofSeq x).IsSt r
    y : Nat → Real
    hys : (Hyperreal.ofSeq y).IsSt s
    ⊢ Hyperreal.IsSt (Filter.Germ.map₂ f (Hyperreal.ofSeq x) (Hyperreal.ofSeq y))  …
  -/
  rw [isSt_ofSeq_iff_tendsto] at hxr hys
  /-
    case intro.intro
    r s : Real
    f : Real → Real → Real
    hf : ContinuousAt (Function.uncurry f) { fst := r, snd := s }
    x : Nat → Real
    hxr : Filter.Tendsto x (↑(Filter.hyperfilter Nat)) (nhds r)
    y : Nat → Real
    hys : Filter.Tendsto y (↑(Filter.hyperfilter Nat)) (nhds s)
    ⊢ Hyperreal.IsSt (Filter.Germ.map₂ f (Hyperreal.ofSeq x) (Hyperreal.ofSeq y))  …
  -/
  exact isSt_ofSeq_iff_tendsto.2 <| hf.tendsto.comp (hxr.prod_mk_nhds hys)
  /-
    🎉 no goals
  -/


theorem IsSt.add {x y : ℝ*} {r s : ℝ} (hxr : IsSt x r) (hys : IsSt y s) :
    IsSt (x + y) (r + s) := hxr.map₂ hys continuous_add.continuousAt


theorem IsSt.neg {x : ℝ*} {r : ℝ} (hxr : IsSt x r) : IsSt (-x) (-r) :=
  hxr.map continuous_neg.continuousAt


theorem IsSt.sub {x y : ℝ*} {r s : ℝ} (hxr : IsSt x r) (hys : IsSt y s) : IsSt (x - y) (r - s) :=
  hxr.map₂ hys continuous_sub.continuousAt


theorem IsSt.le {x y : ℝ*} {r s : ℝ} (hrx : IsSt x r) (hsy : IsSt y s) (hxy : x ≤ y) : r ≤ s :=
  not_lt.1 fun h ↦ hxy.not_lt <| hsy.lt hrx h


theorem st_le_of_le {x y : ℝ*} (hix : ¬Infinite x) (hiy : ¬Infinite y) : x ≤ y → st x ≤ st y :=
  (isSt_st' hix).le (isSt_st' hiy)


theorem lt_of_st_lt {x y : ℝ*} (hix : ¬Infinite x) (hiy : ¬Infinite y) : st x < st y → x < y :=
  (isSt_st' hix).lt (isSt_st' hiy)


theorem infinitePos_def {x : ℝ*} : InfinitePos x ↔ ∀ r : ℝ, ↑r < x := Iff.rfl


theorem infiniteNeg_def {x : ℝ*} : InfiniteNeg x ↔ ∀ r : ℝ, x < r := Iff.rfl


theorem InfinitePos.pos {x : ℝ*} (hip : InfinitePos x) : 0 < x := hip 0


theorem InfiniteNeg.lt_zero {x : ℝ*} : InfiniteNeg x → x < 0 := fun hin => hin 0


theorem Infinite.ne_zero {x : ℝ*} (hI : Infinite x) : x ≠ 0 :=
  hI.elim (fun hip => hip.pos.ne') fun hin => hin.lt_zero.ne


theorem not_infinite_zero : ¬Infinite 0 := fun hI => hI.ne_zero rfl


theorem InfiniteNeg.not_infinitePos {x : ℝ*} : InfiniteNeg x → ¬InfinitePos x := fun hn hp =>
  (hn 0).not_lt (hp 0)


theorem InfinitePos.not_infiniteNeg {x : ℝ*} (hp : InfinitePos x) : ¬InfiniteNeg x := fun hn ↦
  hn.not_infinitePos hp


theorem InfinitePos.neg {x : ℝ*} : InfinitePos x → InfiniteNeg (-x) := fun hp r =>
  neg_lt.mp (hp (-r))


theorem InfiniteNeg.neg {x : ℝ*} : InfiniteNeg x → InfinitePos (-x) := fun hp r =>
  lt_neg.mp (hp (-r))

-- Porting note: swapped LHS with RHS; added @[simp]

@[simp] theorem infiniteNeg_neg {x : ℝ*} : InfiniteNeg (-x) ↔ InfinitePos x :=
  ⟨fun hin => neg_neg x ▸ hin.neg, InfinitePos.neg⟩

-- Porting note: swapped LHS with RHS; added @[simp]

@[simp] theorem infinitePos_neg {x : ℝ*} : InfinitePos (-x) ↔ InfiniteNeg x :=
  ⟨fun hin => neg_neg x ▸ hin.neg, InfiniteNeg.neg⟩

-- Porting note: swapped LHS with RHS; added @[simp]

@[simp] theorem infinite_neg {x : ℝ*} : Infinite (-x) ↔ Infinite x :=
  or_comm.trans <| infiniteNeg_neg.or infinitePos_neg


nonrec theorem Infinitesimal.not_infinite {x : ℝ*} (h : Infinitesimal x) : ¬Infinite x :=
  h.not_infinite


theorem Infinite.not_infinitesimal {x : ℝ*} (h : Infinite x) : ¬Infinitesimal x := fun h' ↦
  h'.not_infinite h


theorem InfinitePos.not_infinitesimal {x : ℝ*} (h : InfinitePos x) : ¬Infinitesimal x :=
  Infinite.not_infinitesimal (Or.inl h)


theorem InfiniteNeg.not_infinitesimal {x : ℝ*} (h : InfiniteNeg x) : ¬Infinitesimal x :=
  Infinite.not_infinitesimal (Or.inr h)


theorem infinitePos_iff_infinite_and_pos {x : ℝ*} : InfinitePos x ↔ Infinite x ∧ 0 < x :=
  ⟨fun hip => ⟨Or.inl hip, hip 0⟩, fun ⟨hi, hp⟩ =>
    hi.casesOn id fun hin => False.elim (not_lt_of_lt hp (hin 0))⟩


theorem infiniteNeg_iff_infinite_and_neg {x : ℝ*} : InfiniteNeg x ↔ Infinite x ∧ x < 0 :=
  ⟨fun hip => ⟨Or.inr hip, hip 0⟩, fun ⟨hi, hp⟩ =>
    hi.casesOn (fun hin => False.elim (not_lt_of_lt hp (hin 0))) fun hip => hip⟩


theorem infinitePos_iff_infinite_of_nonneg {x : ℝ*} (hp : 0 ≤ x) : InfinitePos x ↔ Infinite x :=
  .symm <| or_iff_left fun h ↦ h.lt_zero.not_le hp


theorem infinitePos_iff_infinite_of_pos {x : ℝ*} (hp : 0 < x) : InfinitePos x ↔ Infinite x :=
  infinitePos_iff_infinite_of_nonneg hp.le


theorem infiniteNeg_iff_infinite_of_neg {x : ℝ*} (hn : x < 0) : InfiniteNeg x ↔ Infinite x :=
  .symm <| or_iff_right fun h ↦ h.pos.not_lt hn


theorem infinitePos_abs_iff_infinite_abs {x : ℝ*} : InfinitePos |x| ↔ Infinite |x| :=
  infinitePos_iff_infinite_of_nonneg (abs_nonneg _)

-- Porting note: swapped LHS with RHS; added @[simp]

@[simp] theorem infinite_abs_iff {x : ℝ*} : Infinite |x| ↔ Infinite x := by
  /-
    x : Hyperreal
    ⊢ Iff (abs x).Infinite x.Infinite
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total 0 x <;> simp [*, abs_of_nonneg, abs_of_nonpos, infinite_neg]
                         /-
                           🎉 no goals
                         -/

-- Porting note: swapped LHS with RHS;
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: make it a `simp` lemma

@[simp] theorem infinitePos_abs_iff_infinite {x : ℝ*} : InfinitePos |x| ↔ Infinite x :=
  infinitePos_abs_iff_infinite_abs.trans infinite_abs_iff


theorem infinite_iff_abs_lt_abs {x : ℝ*} : Infinite x ↔ ∀ r : ℝ, (|r| : ℝ*) < |x| :=
  infinitePos_abs_iff_infinite.symm.trans ⟨fun hI r => coe_abs r ▸ hI |r|, fun hR r =>
    (le_abs_self _).trans_lt (hR r)⟩


theorem infinitePos_add_not_infiniteNeg {x y : ℝ*} :
    InfinitePos x → ¬InfiniteNeg y → InfinitePos (x + y) := by
  /-
    x y : Hyperreal
    ⊢ x.InfinitePos → Not y.InfiniteNeg → (HAdd.hAdd x y).InfinitePos
  -/
  intro hip hnin r
  /-
    x y : Hyperreal
    hip : x.InfinitePos
    hnin : Not y.InfiniteNeg
    r : Real
    ⊢ LT.lt (↑r) (HAdd.hAdd x y)
  -/
  cases' not_forall.mp hnin with r₂ hr₂
  /-
    case intro
    x y : Hyperreal
    hip : x.InfinitePos
    hnin : Not y.InfiniteNeg
    r r₂ : Real
    hr₂ : Not (LT.lt y ↑r₂)
    ⊢ LT.lt (↑r) (HAdd.hAdd x y)
  -/
  convert add_lt_add_of_lt_of_le (hip (r + -r₂)) (not_lt.mp hr₂) using 1
  /-
    case h.e'_3
    x y : Hyperreal
    hip : x.InfinitePos
    hnin : Not y.InfiniteNeg
    r r₂ : Real
    hr₂ : Not (LT.lt y ↑r₂)
    ⊢ Eq (↑r) (HAdd.hAdd ↑(HAdd.hAdd r (Neg.neg r₂)) ↑r₂)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem not_infiniteNeg_add_infinitePos {x y : ℝ*} :
    ¬InfiniteNeg x → InfinitePos y → InfinitePos (x + y) := fun hx hy =>
  add_comm y x ▸ infinitePos_add_not_infiniteNeg hy hx


theorem infiniteNeg_add_not_infinitePos {x y : ℝ*} :
    InfiniteNeg x → ¬InfinitePos y → InfiniteNeg (x + y) := by
  /-
    x y : Hyperreal
    ⊢ x.InfiniteNeg → Not y.InfinitePos → (HAdd.hAdd x y).InfiniteNeg
  -/
  rw [← infinitePos_neg, ← infinitePos_neg, ← @infiniteNeg_neg y, neg_add]
  /-
    x y : Hyperreal
    ⊢ (Neg.neg x).InfinitePos → Not (Neg.neg y).InfiniteNeg → (HAdd.hAdd (Neg.neg  …
  -/
  exact infinitePos_add_not_infiniteNeg
  /-
    🎉 no goals
  -/


theorem not_infinitePos_add_infiniteNeg {x y : ℝ*} :
    ¬InfinitePos x → InfiniteNeg y → InfiniteNeg (x + y) := fun hx hy =>
  add_comm y x ▸ infiniteNeg_add_not_infinitePos hy hx


theorem infinitePos_add_infinitePos {x y : ℝ*} :
    InfinitePos x → InfinitePos y → InfinitePos (x + y) := fun hx hy =>
  infinitePos_add_not_infiniteNeg hx hy.not_infiniteNeg


theorem infiniteNeg_add_infiniteNeg {x y : ℝ*} :
    InfiniteNeg x → InfiniteNeg y → InfiniteNeg (x + y) := fun hx hy =>
  infiniteNeg_add_not_infinitePos hx hy.not_infinitePos


theorem infinitePos_add_not_infinite {x y : ℝ*} :
    InfinitePos x → ¬Infinite y → InfinitePos (x + y) := fun hx hy =>
  infinitePos_add_not_infiniteNeg hx (not_or.mp hy).2


theorem infiniteNeg_add_not_infinite {x y : ℝ*} :
    InfiniteNeg x → ¬Infinite y → InfiniteNeg (x + y) := fun hx hy =>
  infiniteNeg_add_not_infinitePos hx (not_or.mp hy).1


theorem infinitePos_of_tendsto_top {f : ℕ → ℝ} (hf : Tendsto f atTop atTop) :
    InfinitePos (ofSeq f) := fun r =>
  have hf' := tendsto_atTop_atTop.mp hf
  let ⟨i, hi⟩ := hf' (r + 1)
  have hi' : ∀ a : ℕ, f a < r + 1 → a < i := fun a => lt_imp_lt_of_le_imp_le (hi a)
  have hS : { a : ℕ | r < f a }ᶜ ⊆ { a : ℕ | a ≤ i } := by
    /-
      f : Nat → Real
      hf : Filter.Tendsto f Filter.atTop Filter.atTop
      r : Real
      hf' : ∀ (b : Real), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (f a)
      i : Nat
      hi : ∀ (a : Nat), LE.le i a → LE.le (HAdd.hAdd r 1) (f a)
      hi' : ∀ (a : Nat), LT.lt (f a) (HAdd.hAdd r 1) → LT.lt a i
      ⊢ HasSubset.Subset (HasCompl.compl (setOf fun a => LT.lt r (f a))) (setOf fun  …
    -/
    simp only [Set.compl_setOf, not_lt]
    /-
      f : Nat → Real
      hf : Filter.Tendsto f Filter.atTop Filter.atTop
      r : Real
      hf' : ∀ (b : Real), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (f a)
      i : Nat
      hi : ∀ (a : Nat), LE.le i a → LE.le (HAdd.hAdd r 1) (f a)
      hi' : ∀ (a : Nat), LT.lt (f a) (HAdd.hAdd r 1) → LT.lt a i
      ⊢ HasSubset.Subset (setOf fun a => LE.le (f a) r) (setOf fun a => LE.le a i)
    -/
    exact fun a har => le_of_lt (hi' a (lt_of_le_of_lt har (lt_add_one _)))
    /-
      🎉 no goals
    -/
  Germ.coe_lt.2 <| mem_hyperfilter_of_finite_compl <| (Set.finite_le_nat _).subset hS


theorem infiniteNeg_of_tendsto_bot {f : ℕ → ℝ} (hf : Tendsto f atTop atBot) :
    InfiniteNeg (ofSeq f) := fun r =>
  have hf' := tendsto_atTop_atBot.mp hf
  let ⟨i, hi⟩ := hf' (r - 1)
  have hi' : ∀ a : ℕ, r - 1 < f a → a < i := fun a => lt_imp_lt_of_le_imp_le (hi a)
  have hS : { a : ℕ | f a < r }ᶜ ⊆ { a : ℕ | a ≤ i } := by
    /-
      f : Nat → Real
      hf : Filter.Tendsto f Filter.atTop Filter.atBot
      r : Real
      hf' : ∀ (b : Real), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (f a) b
      i : Nat
      hi : ∀ (a : Nat), LE.le i a → LE.le (f a) (HSub.hSub r 1)
      hi' : ∀ (a : Nat), LT.lt (HSub.hSub r 1) (f a) → LT.lt a i
      ⊢ HasSubset.Subset (HasCompl.compl (setOf fun a => LT.lt (f a) r)) (setOf fun  …
    -/
    simp only [Set.compl_setOf, not_lt]
    /-
      f : Nat → Real
      hf : Filter.Tendsto f Filter.atTop Filter.atBot
      r : Real
      hf' : ∀ (b : Real), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (f a) b
      i : Nat
      hi : ∀ (a : Nat), LE.le i a → LE.le (f a) (HSub.hSub r 1)
      hi' : ∀ (a : Nat), LT.lt (HSub.hSub r 1) (f a) → LT.lt a i
      ⊢ HasSubset.Subset (setOf fun a => LE.le r (f a)) (setOf fun a => LE.le a i)
    -/
    exact fun a har => le_of_lt (hi' a (lt_of_lt_of_le (sub_one_lt _) har))
    /-
      🎉 no goals
    -/
  Germ.coe_lt.2 <| mem_hyperfilter_of_finite_compl <| (Set.finite_le_nat _).subset hS


theorem not_infinite_neg {x : ℝ*} : ¬Infinite x → ¬Infinite (-x) := mt infinite_neg.mp


theorem not_infinite_add {x y : ℝ*} (hx : ¬Infinite x) (hy : ¬Infinite y) : ¬Infinite (x + y) :=
  have ⟨r, hr⟩ := exists_st_of_not_infinite hx
  have ⟨s, hs⟩ := exists_st_of_not_infinite hy
  not_infinite_of_exists_st <| ⟨r + s, hr.add hs⟩


theorem not_infinite_iff_exist_lt_gt {x : ℝ*} : ¬Infinite x ↔ ∃ r s : ℝ, (r : ℝ*) < x ∧ x < s :=
  ⟨fun hni ↦ let ⟨r, hr⟩ := exists_st_of_not_infinite hni; ⟨r - 1, r + 1, hr 1 one_pos⟩,
    fun ⟨r, s, hr, hs⟩ hi ↦ hi.elim (fun hp ↦ (hp s).not_lt hs) (fun hn ↦ (hn r).not_lt hr)⟩


theorem not_infinite_real (r : ℝ) : ¬Infinite r := by
  /-
    r : Real
    ⊢ Not (↑r).Infinite
  -/
  rw [not_infinite_iff_exist_lt_gt]
  /-
    r : Real
    ⊢ Exists fun r_1 => Exists fun s => And (LT.lt ↑r_1 ↑r) (LT.lt ↑r ↑s)
  -/
  exact ⟨r - 1, r + 1, coe_lt_coe.2 <| sub_one_lt r, coe_lt_coe.2 <| lt_add_one r⟩
  /-
    🎉 no goals
  -/


theorem Infinite.ne_real {x : ℝ*} : Infinite x → ∀ r : ℝ, x ≠ r := fun hi r hr =>
  not_infinite_real r <| @Eq.subst _ Infinite _ _ hr hi


theorem IsSt.mul {x y : ℝ*} {r s : ℝ} (hxr : IsSt x r) (hys : IsSt y s) : IsSt (x * y) (r * s) :=
  hxr.map₂ hys continuous_mul.continuousAt

--AN INFINITE LEMMA THAT REQUIRES SOME MORE ST MACHINERY

theorem not_infinite_mul {x y : ℝ*} (hx : ¬Infinite x) (hy : ¬Infinite y) : ¬Infinite (x * y) :=
  have ⟨_r, hr⟩ := exists_st_of_not_infinite hx
  have ⟨_s, hs⟩ := exists_st_of_not_infinite hy
  (hr.mul hs).not_infinite

---

theorem st_add {x y : ℝ*} (hx : ¬Infinite x) (hy : ¬Infinite y) : st (x + y) = st x + st y :=
  (isSt_st' (not_infinite_add hx hy)).unique ((isSt_st' hx).add (isSt_st' hy))


theorem st_neg (x : ℝ*) : st (-x) = -st x := by
  classical
  by_cases h : Infinite x
  · rw [h.st_eq, (infinite_neg.2 h).st_eq, neg_zero]
  · exact (isSt_st' (not_infinite_neg h)).unique (isSt_st' h).neg


theorem st_mul {x y : ℝ*} (hx : ¬Infinite x) (hy : ¬Infinite y) : st (x * y) = st x * st y :=
  have hx' := isSt_st' hx
  have hy' := isSt_st' hy
  have hxy := isSt_st' (not_infinite_mul hx hy)
  hxy.unique (hx'.mul hy')


theorem infinitesimal_def {x : ℝ*} : Infinitesimal x ↔ ∀ r : ℝ, 0 < r → -(r : ℝ*) < x ∧ x < r := by
  /-
    x : Hyperreal
    ⊢ Iff x.Infinitesimal (∀ (r : Real), LT.lt 0 r → And (LT.lt (Neg.neg ↑r) x) (L …
  -/
  simp [Infinitesimal, IsSt]
  /-
    🎉 no goals
  -/


theorem lt_of_pos_of_infinitesimal {x : ℝ*} : Infinitesimal x → ∀ r : ℝ, 0 < r → x < r :=
  fun hi r hr => ((infinitesimal_def.mp hi) r hr).2


theorem lt_neg_of_pos_of_infinitesimal {x : ℝ*} : Infinitesimal x → ∀ r : ℝ, 0 < r → -↑r < x :=
  fun hi r hr => ((infinitesimal_def.mp hi) r hr).1


theorem gt_of_neg_of_infinitesimal {x : ℝ*} (hi : Infinitesimal x) (r : ℝ) (hr : r < 0) : ↑r < x :=
  neg_neg r ▸ (infinitesimal_def.1 hi (-r) (neg_pos.2 hr)).1


theorem abs_lt_real_iff_infinitesimal {x : ℝ*} : Infinitesimal x ↔ ∀ r : ℝ, r ≠ 0 → |x| < |↑r| :=
  ⟨fun hi r hr ↦ abs_lt.mpr (coe_abs r ▸ infinitesimal_def.mp hi |r| (abs_pos.2 hr)), fun hR ↦
    infinitesimal_def.mpr fun r hr => abs_lt.mp <| (abs_of_pos <| coe_pos.2 hr) ▸ hR r <| hr.ne'⟩


theorem infinitesimal_zero : Infinitesimal 0 := isSt_refl_real 0


theorem Infinitesimal.eq_zero {r : ℝ} : Infinitesimal r → r = 0 := eq_of_isSt_real

-- Porting note: swapped LHS with RHS; added `@[simp]`

@[simp] theorem infinitesimal_real_iff {r : ℝ} : Infinitesimal r ↔ r = 0 :=
  isSt_real_iff_eq


nonrec theorem Infinitesimal.add {x y : ℝ*} (hx : Infinitesimal x) (hy : Infinitesimal y) :
                                /-
                                  x y : Hyperreal
                                  hx : x.Infinitesimal
                                  hy : y.Infinitesimal
                                  ⊢ (HAdd.hAdd x y).Infinitesimal
                                -/
    Infinitesimal (x + y) := by simpa only [add_zero] using hx.add hy
                                /-
                                  🎉 no goals
                                -/


nonrec theorem Infinitesimal.neg {x : ℝ*} (hx : Infinitesimal x) : Infinitesimal (-x) := by
  /-
    x : Hyperreal
    hx : x.Infinitesimal
    ⊢ (Neg.neg x).Infinitesimal
  -/
  simpa only [neg_zero] using hx.neg
  /-
    🎉 no goals
  -/

-- Porting note: swapped LHS and RHS, added `@[simp]`

@[simp] theorem infinitesimal_neg {x : ℝ*} : Infinitesimal (-x) ↔ Infinitesimal x :=
  ⟨fun h => neg_neg x ▸ h.neg, Infinitesimal.neg⟩


nonrec theorem Infinitesimal.mul {x y : ℝ*} (hx : Infinitesimal x) (hy : Infinitesimal y) :
                                /-
                                  x y : Hyperreal
                                  hx : x.Infinitesimal
                                  hy : y.Infinitesimal
                                  ⊢ (HMul.hMul x y).Infinitesimal
                                -/
    Infinitesimal (x * y) := by simpa only [mul_zero] using hx.mul hy
                                /-
                                  🎉 no goals
                                -/


theorem infinitesimal_of_tendsto_zero {f : ℕ → ℝ} (h : Tendsto f atTop (𝓝 0)) :
    Infinitesimal (ofSeq f) :=
  isSt_of_tendsto h


theorem infinitesimal_epsilon : Infinitesimal ε :=
  infinitesimal_of_tendsto_zero tendsto_inverse_atTop_nhds_zero_nat


theorem not_real_of_infinitesimal_ne_zero (x : ℝ*) : Infinitesimal x → x ≠ 0 → ∀ r : ℝ, x ≠ r :=
  fun hi hx r hr =>
  hx <| hr.trans <| coe_eq_zero.2 <| IsSt.unique (hr.symm ▸ isSt_refl_real r : IsSt x r) hi


theorem IsSt.infinitesimal_sub {x : ℝ*} {r : ℝ} (hxr : IsSt x r) : Infinitesimal (x - ↑r) := by
  /-
    x : Hyperreal
    r : Real
    hxr : x.IsSt r
    ⊢ (HSub.hSub x ↑r).Infinitesimal
  -/
  simpa only [sub_self] using hxr.sub (isSt_refl_real r)
  /-
    🎉 no goals
  -/


theorem infinitesimal_sub_st {x : ℝ*} (hx : ¬Infinite x) : Infinitesimal (x - ↑(st x)) :=
  (isSt_st' hx).infinitesimal_sub


theorem infinitePos_iff_infinitesimal_inv_pos {x : ℝ*} :
    InfinitePos x ↔ Infinitesimal x⁻¹ ∧ 0 < x⁻¹ :=
  ⟨fun hip =>
    ⟨infinitesimal_def.mpr fun r hr =>
        ⟨lt_trans (coe_lt_coe.2 (neg_neg_of_pos hr)) (inv_pos.2 (hip 0)),
                                                  /-
                                                    x : Hyperreal
                                                    hip : x.InfinitePos
                                                    r : Real
                                                    hr : LT.lt 0 r
                                                    ⊢ LT.lt (Inv.inv ↑r) x
                                                  -/
          inv_lt_of_inv_lt₀ (coe_lt_coe.2 hr) (by convert hip r⁻¹)⟩,
                                                  /-
                                                    🎉 no goals
                                                  -/
      inv_pos.2 <| hip 0⟩,
    fun ⟨hi, hp⟩ r =>
    @_root_.by_cases (r = 0) (↑r < x) (fun h => Eq.substr h (inv_pos.mp hp)) fun h =>
      lt_of_le_of_lt (coe_le_coe.2 (le_abs_self r))
        ((inv_lt_inv₀ (inv_pos.mp hp) (coe_lt_coe.2 (abs_pos.2 h))).mp
          ((infinitesimal_def.mp hi) |r|⁻¹ (inv_pos.2 (abs_pos.2 h))).2)⟩


theorem infiniteNeg_iff_infinitesimal_inv_neg {x : ℝ*} :
    InfiniteNeg x ↔ Infinitesimal x⁻¹ ∧ x⁻¹ < 0 := by
  /-
    x : Hyperreal
    ⊢ Iff x.InfiniteNeg (And (Inv.inv x).Infinitesimal (LT.lt (Inv.inv x) 0))
  -/
  rw [← infinitePos_neg, infinitePos_iff_infinitesimal_inv_pos, inv_neg, neg_pos, infinitesimal_neg]
  /-
    🎉 no goals
  -/


theorem infinitesimal_inv_of_infinite {x : ℝ*} : Infinite x → Infinitesimal x⁻¹ := fun hi =>
  Or.casesOn hi (fun hip => (infinitePos_iff_infinitesimal_inv_pos.mp hip).1) fun hin =>
    (infiniteNeg_iff_infinitesimal_inv_neg.mp hin).1


theorem infinite_of_infinitesimal_inv {x : ℝ*} (h0 : x ≠ 0) (hi : Infinitesimal x⁻¹) :
    Infinite x := by
  /-
    x : Hyperreal
    h0 : Ne x 0
    hi : (Inv.inv x).Infinitesimal
    ⊢ x.Infinite
  -/
  cases' lt_or_gt_of_ne h0 with hn hp
    /-
      case inl
      x : Hyperreal
      h0 : Ne x 0
      hi : (Inv.inv x).Infinitesimal
      hn : LT.lt x 0
      ⊢ x.Infinite
    -/
  · exact Or.inr (infiniteNeg_iff_infinitesimal_inv_neg.mpr ⟨hi, inv_lt_zero.mpr hn⟩)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Hyperreal
      h0 : Ne x 0
      hi : (Inv.inv x).Infinitesimal
      hp : GT.gt x 0
      ⊢ x.Infinite
    -/
  · exact Or.inl (infinitePos_iff_infinitesimal_inv_pos.mpr ⟨hi, inv_pos.mpr hp⟩)
    /-
      🎉 no goals
    -/


theorem infinite_iff_infinitesimal_inv {x : ℝ*} (h0 : x ≠ 0) : Infinite x ↔ Infinitesimal x⁻¹ :=
  ⟨infinitesimal_inv_of_infinite, infinite_of_infinitesimal_inv h0⟩


theorem infinitesimal_pos_iff_infinitePos_inv {x : ℝ*} :
    InfinitePos x⁻¹ ↔ Infinitesimal x ∧ 0 < x :=
                                                    /-
                                                      x : Hyperreal
                                                      ⊢ Iff (And (Inv.inv (Inv.inv x)).Infinitesimal (LT.lt 0 (Inv.inv (Inv.inv x))) …
                                                    -/
  infinitePos_iff_infinitesimal_inv_pos.trans <| by rw [inv_inv]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem infinitesimal_neg_iff_infiniteNeg_inv {x : ℝ*} :
    InfiniteNeg x⁻¹ ↔ Infinitesimal x ∧ x < 0 :=
                                                    /-
                                                      x : Hyperreal
                                                      ⊢ Iff (And (Inv.inv (Inv.inv x)).Infinitesimal (LT.lt (Inv.inv (Inv.inv x)) 0) …
                                                    -/
  infiniteNeg_iff_infinitesimal_inv_neg.trans <| by rw [inv_inv]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem infinitesimal_iff_infinite_inv {x : ℝ*} (h : x ≠ 0) : Infinitesimal x ↔ Infinite x⁻¹ :=
                /-
                  x : Hyperreal
                  h : Ne x 0
                  ⊢ Iff x.Infinitesimal (Inv.inv (Inv.inv x)).Infinitesimal
                -/
  Iff.trans (by rw [inv_inv]) (infinite_iff_infinitesimal_inv (inv_ne_zero h)).symm
                /-
                  🎉 no goals
                -/


theorem IsSt.inv {x : ℝ*} {r : ℝ} (hi : ¬Infinitesimal x) (hr : IsSt x r) : IsSt x⁻¹ r⁻¹ :=
                                    /-
                                      x : Hyperreal
                                      r : Real
                                      hi : Not x.Infinitesimal
                                      hr : x.IsSt r
                                      ⊢ Ne r 0
                                    -/
  hr.map <| continuousAt_inv₀ <| by rintro rfl; exact hi hr
                                                /-
                                                  🎉 no goals
                                                -/


theorem st_inv (x : ℝ*) : st x⁻¹ = (st x)⁻¹ := by
  /-
    x : Hyperreal
    ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
  -/
  by_cases h0 : x = 0
    /-
      case pos
      x : Hyperreal
      h0 : Eq x 0
      ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
    -/
  · rw [h0, inv_zero, ← coe_zero, st_id_real, inv_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Hyperreal
    h0 : Not (Eq x 0)
    ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
  -/
  by_cases h1 : Infinitesimal x
    /-
      case pos
      x : Hyperreal
      h0 : Not (Eq x 0)
      h1 : x.Infinitesimal
      ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
    -/
  · rw [((infinitesimal_iff_infinite_inv h0).mp h1).st_eq, h1.st_eq, inv_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Hyperreal
    h0 : Not (Eq x 0)
    h1 : Not x.Infinitesimal
    ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
  -/
  by_cases h2 : Infinite x
    /-
      case pos
      x : Hyperreal
      h0 : Not (Eq x 0)
      h1 : Not x.Infinitesimal
      h2 : x.Infinite
      ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
    -/
  · rw [(infinitesimal_inv_of_infinite h2).st_eq, h2.st_eq, inv_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Hyperreal
    h0 : Not (Eq x 0)
    h1 : Not x.Infinitesimal
    h2 : Not x.Infinite
    ⊢ Eq (Inv.inv x).st (Inv.inv x.st)
  -/
  exact ((isSt_st' h2).inv h1).st_eq
  /-
    🎉 no goals
  -/


theorem infinitePos_omega : InfinitePos ω :=
  infinitePos_iff_infinitesimal_inv_pos.mpr ⟨infinitesimal_epsilon, epsilon_pos⟩


theorem infinite_omega : Infinite ω :=
  (infinite_iff_infinitesimal_inv omega_ne_zero).mpr infinitesimal_epsilon


theorem infinitePos_mul_of_infinitePos_not_infinitesimal_pos {x y : ℝ*} :
    InfinitePos x → ¬Infinitesimal y → 0 < y → InfinitePos (x * y) := fun hx hy₁ hy₂ r => by
  /-
    x y : Hyperreal
    hx : x.InfinitePos
    hy₁ : Not y.Infinitesimal
    hy₂ : LT.lt 0 y
    r : Real
    ⊢ LT.lt (↑r) (HMul.hMul x y)
  -/
  have hy₁' := not_forall.mp (mt infinitesimal_def.2 hy₁)
  /-
    x y : Hyperreal
    hx : x.InfinitePos
    hy₁ : Not y.Infinitesimal
    hy₂ : LT.lt 0 y
    r : Real
    hy₁' : Exists fun x => Not (LT.lt 0 x → And (LT.lt (Neg.neg ↑x) y) (LT.lt y ↑x))
    ⊢ LT.lt (↑r) (HMul.hMul x y)
  -/
  let ⟨r₁, hy₁''⟩ := hy₁'
  have hyr : 0 < r₁ ∧ ↑r₁ ≤ y := by
    rwa [Classical.not_imp, ← abs_lt, not_lt, abs_of_pos hy₂] at hy₁''
  /-
    x y : Hyperreal
    hx : x.InfinitePos
    hy₁ : Not y.Infinitesimal
    hy₂ : LT.lt 0 y
    r : Real
    hy₁' : Exists fun x => Not (LT.lt 0 x → And (LT.lt (Neg.neg ↑x) y) (LT.lt y ↑x))
    r₁ : Real
    hy₁'' : Not (LT.lt 0 r₁ → And (LT.lt (Neg.neg ↑r₁) y) (LT.lt y ↑r₁))
    hyr : And (LT.lt 0 r₁) (LE.le (↑r₁) y)
    ⊢ LT.lt (↑r) (HMul.hMul x y)
  -/
  rw [← div_mul_cancel₀ r (ne_of_gt hyr.1), coe_mul]
  /-
    x y : Hyperreal
    hx : x.InfinitePos
    hy₁ : Not y.Infinitesimal
    hy₂ : LT.lt 0 y
    r : Real
    hy₁' : Exists fun x => Not (LT.lt 0 x → And (LT.lt (Neg.neg ↑x) y) (LT.lt y ↑x))
    r₁ : Real
    hy₁'' : Not (LT.lt 0 r₁ → And (LT.lt (Neg.neg ↑r₁) y) (LT.lt y ↑r₁))
    hyr : And (LT.lt 0 r₁) (LE.le (↑r₁) y)
    ⊢ LT.lt (HMul.hMul ↑(HDiv.hDiv r r₁) ↑r₁) (HMul.hMul x y)
  -/
  exact mul_lt_mul (hx (r / r₁)) hyr.2 (coe_lt_coe.2 hyr.1) (le_of_lt (hx 0))
  /-
    🎉 no goals
  -/


theorem infinitePos_mul_of_not_infinitesimal_pos_infinitePos {x y : ℝ*} :
    ¬Infinitesimal x → 0 < x → InfinitePos y → InfinitePos (x * y) := fun hx hp hy =>
  mul_comm y x ▸ infinitePos_mul_of_infinitePos_not_infinitesimal_pos hy hx hp


theorem infinitePos_mul_of_infiniteNeg_not_infinitesimal_neg {x y : ℝ*} :
    InfiniteNeg x → ¬Infinitesimal y → y < 0 → InfinitePos (x * y) := by
  /-
    x y : Hyperreal
    ⊢ x.InfiniteNeg → Not y.Infinitesimal → LT.lt y 0 → (HMul.hMul x y).InfinitePos
  -/
  rw [← infinitePos_neg, ← neg_pos, ← neg_mul_neg, ← infinitesimal_neg]
  /-
    x y : Hyperreal
    ⊢ (Neg.neg x).InfinitePos → Not (Neg.neg y).Infinitesimal → LT.lt 0 (Neg.neg y …
  -/
  exact infinitePos_mul_of_infinitePos_not_infinitesimal_pos
  /-
    🎉 no goals
  -/


theorem infinitePos_mul_of_not_infinitesimal_neg_infiniteNeg {x y : ℝ*} :
    ¬Infinitesimal x → x < 0 → InfiniteNeg y → InfinitePos (x * y) := fun hx hp hy =>
  mul_comm y x ▸ infinitePos_mul_of_infiniteNeg_not_infinitesimal_neg hy hx hp


theorem infiniteNeg_mul_of_infinitePos_not_infinitesimal_neg {x y : ℝ*} :
    InfinitePos x → ¬Infinitesimal y → y < 0 → InfiniteNeg (x * y) := by
  /-
    x y : Hyperreal
    ⊢ x.InfinitePos → Not y.Infinitesimal → LT.lt y 0 → (HMul.hMul x y).InfiniteNeg
  -/
  rw [← infinitePos_neg, ← neg_pos, neg_mul_eq_mul_neg, ← infinitesimal_neg]
  /-
    x y : Hyperreal
    ⊢ x.InfinitePos → Not (Neg.neg y).Infinitesimal → LT.lt 0 (Neg.neg y) → (HMul. …
  -/
  exact infinitePos_mul_of_infinitePos_not_infinitesimal_pos
  /-
    🎉 no goals
  -/


theorem infiniteNeg_mul_of_not_infinitesimal_neg_infinitePos {x y : ℝ*} :
    ¬Infinitesimal x → x < 0 → InfinitePos y → InfiniteNeg (x * y) := fun hx hp hy =>
  mul_comm y x ▸ infiniteNeg_mul_of_infinitePos_not_infinitesimal_neg hy hx hp


theorem infiniteNeg_mul_of_infiniteNeg_not_infinitesimal_pos {x y : ℝ*} :
    InfiniteNeg x → ¬Infinitesimal y → 0 < y → InfiniteNeg (x * y) := by
  /-
    x y : Hyperreal
    ⊢ x.InfiniteNeg → Not y.Infinitesimal → LT.lt 0 y → (HMul.hMul x y).InfiniteNeg
  -/
  rw [← infinitePos_neg, ← infinitePos_neg, neg_mul_eq_neg_mul]
  /-
    x y : Hyperreal
    ⊢ (Neg.neg x).InfinitePos → Not y.Infinitesimal → LT.lt 0 y → (HMul.hMul (Neg. …
  -/
  exact infinitePos_mul_of_infinitePos_not_infinitesimal_pos
  /-
    🎉 no goals
  -/


theorem infiniteNeg_mul_of_not_infinitesimal_pos_infiniteNeg {x y : ℝ*} :
    ¬Infinitesimal x → 0 < x → InfiniteNeg y → InfiniteNeg (x * y) := fun hx hp hy => by
  /-
    x y : Hyperreal
    hx : Not x.Infinitesimal
    hp : LT.lt 0 x
    hy : y.InfiniteNeg
    ⊢ (HMul.hMul x y).InfiniteNeg
  -/
  rw [mul_comm]; exact infiniteNeg_mul_of_infiniteNeg_not_infinitesimal_pos hy hx hp
                 /-
                   🎉 no goals
                 -/


theorem infinitePos_mul_infinitePos {x y : ℝ*} :
    InfinitePos x → InfinitePos y → InfinitePos (x * y) := fun hx hy =>
  infinitePos_mul_of_infinitePos_not_infinitesimal_pos hx hy.not_infinitesimal (hy 0)


theorem infiniteNeg_mul_infiniteNeg {x y : ℝ*} :
    InfiniteNeg x → InfiniteNeg y → InfinitePos (x * y) := fun hx hy =>
  infinitePos_mul_of_infiniteNeg_not_infinitesimal_neg hx hy.not_infinitesimal (hy 0)


theorem infinitePos_mul_infiniteNeg {x y : ℝ*} :
    InfinitePos x → InfiniteNeg y → InfiniteNeg (x * y) := fun hx hy =>
  infiniteNeg_mul_of_infinitePos_not_infinitesimal_neg hx hy.not_infinitesimal (hy 0)


theorem infiniteNeg_mul_infinitePos {x y : ℝ*} :
    InfiniteNeg x → InfinitePos y → InfiniteNeg (x * y) := fun hx hy =>
  infiniteNeg_mul_of_infiniteNeg_not_infinitesimal_pos hx hy.not_infinitesimal (hy 0)


theorem infinite_mul_of_infinite_not_infinitesimal {x y : ℝ*} :
    Infinite x → ¬Infinitesimal y → Infinite (x * y) := fun hx hy =>
  have h0 : y < 0 ∨ 0 < y := lt_or_gt_of_ne fun H0 => hy (Eq.substr H0 (isSt_refl_real 0))
  hx.elim
    (h0.elim
      (fun H0 Hx => Or.inr (infiniteNeg_mul_of_infinitePos_not_infinitesimal_neg Hx hy H0))
      fun H0 Hx => Or.inl (infinitePos_mul_of_infinitePos_not_infinitesimal_pos Hx hy H0))
    (h0.elim
      (fun H0 Hx => Or.inl (infinitePos_mul_of_infiniteNeg_not_infinitesimal_neg Hx hy H0))
      fun H0 Hx => Or.inr (infiniteNeg_mul_of_infiniteNeg_not_infinitesimal_pos Hx hy H0))


theorem infinite_mul_of_not_infinitesimal_infinite {x y : ℝ*} :
    ¬Infinitesimal x → Infinite y → Infinite (x * y) := fun hx hy => by
  /-
    x y : Hyperreal
    hx : Not x.Infinitesimal
    hy : y.Infinite
    ⊢ (HMul.hMul x y).Infinite
  -/
  rw [mul_comm]; exact infinite_mul_of_infinite_not_infinitesimal hy hx
                 /-
                   🎉 no goals
                 -/


theorem Infinite.mul {x y : ℝ*} : Infinite x → Infinite y → Infinite (x * y) := fun hx hy =>
  infinite_mul_of_infinite_not_infinitesimal hx hy.not_infinitesimal


