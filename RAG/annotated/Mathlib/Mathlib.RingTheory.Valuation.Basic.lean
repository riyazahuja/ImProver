/-- The type of `Γ₀`-valued valuations on `R`.

When you extend this structure, make sure to extend `ValuationClass`. -/
structure Valuation extends R →*₀ Γ₀ where
  /-- The valuation of a a sum is less that the sum of the valuations -/
  map_add_le_max' : ∀ x y, toFun (x + y) ≤ max (toFun x) (toFun y)


/-- `ValuationClass F α β` states that `F` is a type of valuations.

You should also extend this typeclass when you extend `Valuation`. -/
class ValuationClass (F) (R Γ₀ : outParam Type*) [LinearOrderedCommMonoidWithZero Γ₀] [Ring R]
  [FunLike F R Γ₀]
  extends MonoidWithZeroHomClass F R Γ₀ : Prop where
  /-- The valuation of a a sum is less that the sum of the valuations -/
  map_add_le_max (f : F) (x y : R) : f (x + y) ≤ max (f x) (f y)


instance [FunLike F R Γ₀] [ValuationClass F R Γ₀] : CoeTC F (Valuation R Γ₀) :=
  ⟨fun f =>
    { toFun := f
      map_one' := map_one f
      map_zero' := map_zero f
      map_mul' := map_mul f
      map_add_le_max' := map_add_le_max f }⟩


instance : FunLike (Valuation R Γ₀) R Γ₀ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      K : Type u_1
      F : Type u_2
      R : Type u_3
      inst✝⁴ : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      Γ''₀ : Type u_6
      inst✝³ : LinearOrderedCommMonoidWithZero Γ''₀
      inst✝² : Ring R
      inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
      inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
      f g : Valuation R Γ₀
      h : Eq ((fun f => (↑f.toMonoidWithZeroHom).toFun) f) ((fun f => (↑f.toMonoidWi …
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨_,_⟩, _⟩, _⟩ := f
    /-
      case mk.mk.mk
      K : Type u_1
      F : Type u_2
      R : Type u_3
      inst✝⁴ : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      Γ''₀ : Type u_6
      inst✝³ : LinearOrderedCommMonoidWithZero Γ''₀
      inst✝² : Ring R
      inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
      inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
      g : Valuation R Γ₀
      toFun✝ : R → Γ₀
      map_zero'✝ : Eq (toFun✝ 0) 0
      map_one'✝ : Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFun 1) 1
      map_mul'✝ : ∀ (x y : R), Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFu …
      map_add_le_max'✝ : ∀ (x y : R), LE.le ((↑{ toFun := toFun✝, map_zero' := map_z …
      h : Eq ((fun f => (↑f.toMonoidWithZeroHom).toFun) { toFun := toFun✝, map_zero' …
      ⊢ Eq { toFun := toFun✝, map_zero' := map_zero'✝, map_one' := map_one'✝, map_mu …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : ValuationClass (Valuation R Γ₀) R Γ₀ where
  map_mul f := f.map_mul'
  map_one f := f.map_one'
  map_zero f := f.map_zero'
  map_add_le_max f := f.map_add_le_max'


@[simp]
theorem coe_mk (f : R →*₀ Γ₀) (h) : ⇑(Valuation.mk f h) = f := rfl


theorem toFun_eq_coe (v : Valuation R Γ₀) : v.toFun = v := rfl


@[simp]
theorem toMonoidWithZeroHom_coe_eq_coe (v : Valuation R Γ₀) :
    (v.toMonoidWithZeroHom : R → Γ₀) = v := rfl


@[ext]
theorem ext {v₁ v₂ : Valuation R Γ₀} (h : ∀ r, v₁ r = v₂ r) : v₁ = v₂ :=
  DFunLike.ext _ _ h


@[simp, norm_cast]
theorem coe_coe : ⇑(v : R →*₀ Γ₀) = v := rfl


theorem map_zero : v 0 = 0 :=
  v.map_zero'


theorem map_one : v 1 = 1 :=
  v.map_one'


theorem map_mul : ∀ x y, v (x * y) = v x * v y :=
  v.map_mul'

-- Porting note: LHS side simplified so created map_add'

theorem map_add : ∀ x y, v (x + y) ≤ max (v x) (v y) :=
  v.map_add_le_max'


@[simp]
theorem map_add' : ∀ x y, v (x + y) ≤ v x ∨ v (x + y) ≤ v y := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ⊢ ∀ (x y : R), Or (LE.le (v (HAdd.hAdd x y)) (v x)) (LE.le (v (HAdd.hAdd x y)) …
  -/
  intro x y
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    ⊢ Or (LE.le (v (HAdd.hAdd x y)) (v x)) (LE.le (v (HAdd.hAdd x y)) (v y))
  -/
  rw [← le_max_iff, ← ge_iff_le]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    ⊢ GE.ge (Max.max (v x) (v y)) (v (HAdd.hAdd x y))
  -/
  apply map_add
  /-
    🎉 no goals
  -/


theorem map_add_le {x y g} (hx : v x ≤ g) (hy : v y ≤ g) : v (x + y) ≤ g :=
  le_trans (v.map_add x y) <| max_le hx hy


theorem map_add_lt {x y g} (hx : v x < g) (hy : v y < g) : v (x + y) < g :=
  lt_of_le_of_lt (v.map_add x y) <| max_lt hx hy


theorem map_sum_le {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hf : ∀ i ∈ s, v (f i) ≤ g) :
    v (∑ i ∈ s, f i) ≤ g := by
  refine
    Finset.induction_on s (fun _ => v.map_zero ▸ zero_le')
      (fun a s has ih hf => ?_) hf
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s✝ : Finset ι
    f : ι → R
    g : Γ₀
    hf✝ : ∀ (i : ι), Membership.mem s✝ i → LE.le (v (f i)) g
    a : ι
    s : Finset ι
    has : Not (Membership.mem s a)
    ih : (∀ (i : ι), Membership.mem s i → LE.le (v (f i)) g) → LE.le (v (s.sum fun …
    hf : ∀ (i : ι), Membership.mem (Insert.insert a s) i → LE.le (v (f i)) g
    ⊢ LE.le (v ((Insert.insert a s).sum fun i => f i)) g
  -/
  rw [Finset.forall_mem_insert] at hf; rw [Finset.sum_insert has]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s✝ : Finset ι
    f : ι → R
    g : Γ₀
    hf✝ : ∀ (i : ι), Membership.mem s✝ i → LE.le (v (f i)) g
    a : ι
    s : Finset ι
    has : Not (Membership.mem s a)
    ih : (∀ (i : ι), Membership.mem s i → LE.le (v (f i)) g) → LE.le (v (s.sum fun …
    hf : And (LE.le (v (f a)) g) (∀ (x : ι), Membership.mem s x → LE.le (v (f x)) g)
    ⊢ LE.le (v (HAdd.hAdd (f a) (s.sum fun x => f x))) g
  -/
  exact v.map_add_le hf.1 (ih hf.2)
  /-
    🎉 no goals
  -/


theorem map_sum_lt {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hg : g ≠ 0)
    (hf : ∀ i ∈ s, v (f i) < g) : v (∑ i ∈ s, f i) < g := by
  refine
    Finset.induction_on s (fun _ => v.map_zero ▸ (zero_lt_iff.2 hg))
      (fun a s has ih hf => ?_) hf
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s✝ : Finset ι
    f : ι → R
    g : Γ₀
    hg : Ne g 0
    hf✝ : ∀ (i : ι), Membership.mem s✝ i → LT.lt (v (f i)) g
    a : ι
    s : Finset ι
    has : Not (Membership.mem s a)
    ih : (∀ (i : ι), Membership.mem s i → LT.lt (v (f i)) g) → LT.lt (v (s.sum fun …
    hf : ∀ (i : ι), Membership.mem (Insert.insert a s) i → LT.lt (v (f i)) g
    ⊢ LT.lt (v ((Insert.insert a s).sum fun i => f i)) g
  -/
  rw [Finset.forall_mem_insert] at hf; rw [Finset.sum_insert has]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s✝ : Finset ι
    f : ι → R
    g : Γ₀
    hg : Ne g 0
    hf✝ : ∀ (i : ι), Membership.mem s✝ i → LT.lt (v (f i)) g
    a : ι
    s : Finset ι
    has : Not (Membership.mem s a)
    ih : (∀ (i : ι), Membership.mem s i → LT.lt (v (f i)) g) → LT.lt (v (s.sum fun …
    hf : And (LT.lt (v (f a)) g) (∀ (x : ι), Membership.mem s x → LT.lt (v (f x)) g)
    ⊢ LT.lt (v (HAdd.hAdd (f a) (s.sum fun x => f x))) g
  -/
  exact v.map_add_lt hf.1 (ih hf.2)
  /-
    🎉 no goals
  -/


theorem map_sum_lt' {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hg : 0 < g)
    (hf : ∀ i ∈ s, v (f i) < g) : v (∑ i ∈ s, f i) < g :=
  v.map_sum_lt (ne_of_gt hg) hf


theorem map_pow : ∀ (x) (n : ℕ), v (x ^ n) = v x ^ n :=
  v.toMonoidWithZeroHom.toMonoidHom.map_pow

-- The following definition is not an instance, because we have more than one `v` on a given `R`.
-- In addition, type class inference would not be able to infer `v`.

/-- A valuation gives a preorder on the underlying ring. -/
def toPreorder : Preorder R :=
  Preorder.lift v


/-- If `v` is a valuation on a division ring then `v(x) = 0` iff `x = 0`. -/
theorem zero_iff [Nontrivial Γ₀] (v : Valuation K Γ₀) {x : K} : v x = 0 ↔ x = 0 :=
  map_eq_zero v


theorem ne_zero_iff [Nontrivial Γ₀] (v : Valuation K Γ₀) {x : K} : v x ≠ 0 ↔ x ≠ 0 :=
  map_ne_zero v


lemma pos_iff [Nontrivial Γ₀] (v : Valuation K Γ₀) {x : K} : 0 < v x ↔ x ≠ 0 := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : Nontrivial Γ₀
    v : Valuation K Γ₀
    x : K
    ⊢ Iff (LT.lt 0 (v x)) (Ne x 0)
  -/
  rw [zero_lt_iff, ne_zero_iff]
  /-
    🎉 no goals
  -/


theorem unit_map_eq (u : Rˣ) : (Units.map (v : R →* Γ₀) u : Γ₀) = v u :=
  rfl


theorem ne_zero_of_unit [Nontrivial Γ₀] (v : Valuation K Γ₀) (x : Kˣ) : v x ≠ (0 : Γ₀) := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : Nontrivial Γ₀
    v : Valuation K Γ₀
    x : Units K
    ⊢ Ne (v ↑x) 0
  -/
  simp only [ne_eq, Valuation.zero_iff, Units.ne_zero x, not_false_iff]
  /-
    🎉 no goals
  -/


theorem ne_zero_of_isUnit [Nontrivial Γ₀] (v : Valuation K Γ₀) (x : K) (hx : IsUnit x) :
    v x ≠ (0 : Γ₀) := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : Nontrivial Γ₀
    v : Valuation K Γ₀
    x : K
    hx : IsUnit x
    ⊢ Ne (v x) 0
  -/
  simpa [hx.choose_spec] using ne_zero_of_unit v hx.choose
  /-
    🎉 no goals
  -/


/-- A ring homomorphism `S → R` induces a map `Valuation R Γ₀ → Valuation S Γ₀`. -/
def comap {S : Type*} [Ring S] (f : S →+* R) (v : Valuation R Γ₀) : Valuation S Γ₀ :=
  { v.toMonoidWithZeroHom.comp f.toMonoidWithZeroHom with
    toFun := v ∘ f
                                     /-
                                       K : Type u_1
                                       F : Type u_2
                                       R : Type u_3
                                       inst✝⁵ : DivisionRing K
                                       Γ₀ : Type u_4
                                       Γ'₀ : Type u_5
                                       Γ''₀ : Type u_6
                                       inst✝⁴ : LinearOrderedCommMonoidWithZero Γ''₀
                                       inst✝³ : Ring R
                                       inst✝² : LinearOrderedCommMonoidWithZero Γ₀
                                       inst✝¹ : LinearOrderedCommMonoidWithZero Γ'₀
                                       v✝ : Valuation R Γ₀
                                       S : Type u_7
                                       inst✝ : Ring S
                                       f : RingHom S R
                                       v : Valuation R Γ₀
                                       x y : S
                                       ⊢ LE.le ((↑{ toFun := Function.comp ⇑v ⇑f, map_zero' := ⋯, map_one' := ⋯, map_ …
                                     -/
    map_add_le_max' := fun x y => by simp only [comp_apply, map_add, f.map_add] }
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem comap_apply {S : Type*} [Ring S] (f : S →+* R) (v : Valuation R Γ₀) (s : S) :
    v.comap f s = v (f s) := rfl


@[simp]
theorem comap_id : v.comap (RingHom.id R) = v :=
  ext fun _r => rfl


theorem comap_comp {S₁ : Type*} {S₂ : Type*} [Ring S₁] [Ring S₂] (f : S₁ →+* S₂) (g : S₂ →+* R) :
    v.comap (g.comp f) = (v.comap g).comap f :=
  ext fun _r => rfl


/-- A `≤`-preserving group homomorphism `Γ₀ → Γ'₀` induces a map `Valuation R Γ₀ → Valuation R Γ'₀`.
-/
def map (f : Γ₀ →*₀ Γ'₀) (hf : Monotone f) (v : Valuation R Γ₀) : Valuation R Γ'₀ :=
  { MonoidWithZeroHom.comp f v.toMonoidWithZeroHom with
    toFun := f ∘ v
    map_add_le_max' := fun r s =>
      calc
        f (v (r + s)) ≤ f (max (v r) (v s)) := hf (v.map_add r s)
        _ = max (f (v r)) (f (v s)) := hf.map_max
         }


@[simp]
lemma map_apply (f : Γ₀ →*₀ Γ'₀) (hf : Monotone f) (v : Valuation R Γ₀) (r : R) :
    v.map f hf r = f (v r) := rfl


/-- Two valuations on `R` are defined to be equivalent if they induce the same preorder on `R`. -/
def IsEquiv (v₁ : Valuation R Γ₀) (v₂ : Valuation R Γ'₀) : Prop :=
  ∀ r s, v₁ r ≤ v₁ s ↔ v₂ r ≤ v₂ s


@[simp]
theorem map_neg (x : R) : v (-x) = v x :=
  v.toMonoidWithZeroHom.toMonoidHom.map_neg x


theorem map_sub_swap (x y : R) : v (x - y) = v (y - x) :=
  v.toMonoidWithZeroHom.toMonoidHom.map_sub_swap x y


theorem map_inv {R : Type*} [DivisionRing R] (v : Valuation R Γ₀) : ∀ x, v x⁻¹ = (v x)⁻¹ :=
  map_inv₀ _


theorem map_div {R : Type*} [DivisionRing R] (v : Valuation R Γ₀) : ∀ x y, v (x / y) = v x / v y :=
  map_div₀ _


theorem map_sub (x y : R) : v (x - y) ≤ max (v x) (v y) :=
  calc
                                 /-
                                   R : Type u_3
                                   Γ₀ : Type u_4
                                   inst✝¹ : Ring R
                                   inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                   v : Valuation R Γ₀
                                   x y : R
                                   ⊢ Eq (v (HSub.hSub x y)) (v (HAdd.hAdd x (Neg.neg y)))
                                 -/
    v (x - y) = v (x + -y) := by rw [sub_eq_add_neg]
                                 /-
                                   🎉 no goals
                                 -/
    _ ≤ max (v x) (v <| -y) := v.map_add _ _
                              /-
                                R : Type u_3
                                Γ₀ : Type u_4
                                inst✝¹ : Ring R
                                inst✝ : LinearOrderedCommGroupWithZero Γ₀
                                v : Valuation R Γ₀
                                x y : R
                                ⊢ Eq (Max.max (v x) (v (Neg.neg y))) (Max.max (v x) (v y))
                              -/
    _ = max (v x) (v y) := by rw [map_neg]
                              /-
                                🎉 no goals
                              -/


theorem map_sub_le {x y g} (hx : v x ≤ g) (hy : v y ≤ g) : v (x - y) ≤ g := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    g : Γ₀
    hx : LE.le (v x) g
    hy : LE.le (v y) g
    ⊢ LE.le (v (HSub.hSub x y)) g
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    g : Γ₀
    hx : LE.le (v x) g
    hy : LE.le (v y) g
    ⊢ LE.le (v (HAdd.hAdd x (Neg.neg y))) g
  -/
  exact v.map_add_le hx (le_trans (le_of_eq (v.map_neg y)) hy)
  /-
    🎉 no goals
  -/


theorem map_add_of_distinct_val (h : v x ≠ v y) : v (x + y) = max (v x) (v y) := by
  suffices ¬v (x + y) < max (v x) (v y) from
    or_iff_not_imp_right.1 (le_iff_eq_or_lt.1 (v.map_add x y)) this
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : Ne (v x) (v y)
    ⊢ Not (LT.lt (v (HAdd.hAdd x y)) (Max.max (v x) (v y)))
  -/
  intro h'
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : Ne (v x) (v y)
    h' : LT.lt (v (HAdd.hAdd x y)) (Max.max (v x) (v y))
    ⊢ False
  -/
  wlog vyx : v y < v x generalizing x y
    /-
      case inr
      R : Type u_3
      Γ₀ : Type u_4
      inst✝¹ : Ring R
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      x y : R
      h : Ne (v x) (v y)
      h' : LT.lt (v (HAdd.hAdd x y)) (Max.max (v x) (v y))
      this : ∀ {x y : R}, Ne (v x) (v y) → LT.lt (v (HAdd.hAdd x y)) (Max.max (v x)  …
      vyx : Not (LT.lt (v y) (v x))
      ⊢ False
    -/
  · refine this h.symm ?_ (h.lt_or_lt.resolve_right vyx)
    /-
      case inr
      R : Type u_3
      Γ₀ : Type u_4
      inst✝¹ : Ring R
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      x y : R
      h : Ne (v x) (v y)
      h' : LT.lt (v (HAdd.hAdd x y)) (Max.max (v x) (v y))
      this : ∀ {x y : R}, Ne (v x) (v y) → LT.lt (v (HAdd.hAdd x y)) (Max.max (v x)  …
      vyx : Not (LT.lt (v y) (v x))
      ⊢ LT.lt (v (HAdd.hAdd y x)) (Max.max (v y) (v x))
    -/
    rwa [add_comm, max_comm]
    /-
      🎉 no goals
    -/
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x✝ y✝ x y : R
    h : Ne (v x) (v y)
    h' : LT.lt (v (HAdd.hAdd x y)) (Max.max (v x) (v y))
    vyx : LT.lt (v y) (v x)
    ⊢ False
  -/
  rw [max_eq_left_of_lt vyx] at h'
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x✝ y✝ x y : R
    h : Ne (v x) (v y)
    h' : LT.lt (v (HAdd.hAdd x y)) (v x)
    vyx : LT.lt (v y) (v x)
    ⊢ False
  -/
  apply lt_irrefl (v x)
  calc
    v x = v (x + y - y) := by simp
    _ ≤ max (v <| x + y) (v y) := map_sub _ _ _
    _ < v x := max_lt h' vyx


theorem map_add_eq_of_lt_right (h : v x < v y) : v (x + y) = v y :=
  (v.map_add_of_distinct_val h.ne).trans (max_eq_right_iff.mpr h.le)


theorem map_add_eq_of_lt_left (h : v y < v x) : v (x + y) = v x := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v y) (v x)
    ⊢ Eq (v (HAdd.hAdd x y)) (v x)
  -/
  rw [add_comm]; exact map_add_eq_of_lt_right _ h
                 /-
                   🎉 no goals
                 -/


theorem map_sub_eq_of_lt_right (h : v x < v y) : v (x - y) = v y := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v x) (v y)
    ⊢ Eq (v (HSub.hSub x y)) (v y)
  -/
  rw [sub_eq_add_neg, map_add_eq_of_lt_right, map_neg]
  /-
    case h
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v x) (v y)
    ⊢ LT.lt (v x) (v (Neg.neg y))
  -/
  rwa [map_neg]
  /-
    🎉 no goals
  -/


theorem map_sum_eq_of_lt {ι : Type*} {s : Finset ι} {f : ι → R} {j : ι}
    (hj : j ∈ s) (h0 : v (f j) ≠ 0) (hf : ∀ i ∈ s \ {j}, v (f i) < v (f j)) :
    v (∑ i ∈ s, f i) = v (f j) := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s : Finset ι
    f : ι → R
    j : ι
    hj : Membership.mem s j
    h0 : Ne (v (f j)) 0
    hf : ∀ (i : ι), Membership.mem (SDiff.sdiff s (Singleton.singleton j)) i → LT. …
    ⊢ Eq (v (s.sum fun i => f i)) (v (f j))
  -/
  rw [Finset.sum_eq_add_sum_diff_singleton hj]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    ι : Type u_7
    s : Finset ι
    f : ι → R
    j : ι
    hj : Membership.mem s j
    h0 : Ne (v (f j)) 0
    hf : ∀ (i : ι), Membership.mem (SDiff.sdiff s (Singleton.singleton j)) i → LT. …
    ⊢ Eq (v (HAdd.hAdd (f j) ((SDiff.sdiff s (Singleton.singleton j)).sum fun x => …
  -/
  exact map_add_eq_of_lt_left _ (map_sum_lt _ h0 hf)
  /-
    🎉 no goals
  -/


theorem map_sub_eq_of_lt_left (h : v y < v x) : v (x - y) = v x := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v y) (v x)
    ⊢ Eq (v (HSub.hSub x y)) (v x)
  -/
  rw [sub_eq_add_neg, map_add_eq_of_lt_left]
  /-
    case h
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v y) (v x)
    ⊢ LT.lt (v (Neg.neg y)) (v x)
  -/
  rwa [map_neg]
  /-
    🎉 no goals
  -/


theorem map_eq_of_sub_lt (h : v (y - x) < v x) : v y = v x := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v (HSub.hSub y x)) (v x)
    ⊢ Eq (v y) (v x)
  -/
  have := Valuation.map_add_of_distinct_val v (ne_of_gt h).symm
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v (HSub.hSub y x)) (v x)
    this : Eq (v (HAdd.hAdd (HSub.hSub y x) x)) (Max.max (v (HSub.hSub y x)) (v x))
    ⊢ Eq (v y) (v x)
  -/
  rw [max_eq_right (le_of_lt h)] at this
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x y : R
    h : LT.lt (v (HSub.hSub y x)) (v x)
    this : Eq (v (HAdd.hAdd (HSub.hSub y x) x)) (v x)
    ⊢ Eq (v y) (v x)
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem map_one_add_of_lt (h : v x < 1) : v (1 + x) = 1 := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x : R
    h : LT.lt (v x) 1
    ⊢ Eq (v (HAdd.hAdd 1 x)) 1
  -/
  rw [← v.map_one] at h
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x : R
    h : LT.lt (v x) (v 1)
    ⊢ Eq (v (HAdd.hAdd 1 x)) 1
  -/
  simpa only [v.map_one] using v.map_add_eq_of_lt_left h
  /-
    🎉 no goals
  -/


theorem map_one_sub_of_lt (h : v x < 1) : v (1 - x) = 1 := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x : R
    h : LT.lt (v x) 1
    ⊢ Eq (v (HSub.hSub 1 x)) 1
  -/
  rw [← v.map_one, ← v.map_neg] at h
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x : R
    h : LT.lt (v (Neg.neg x)) (v 1)
    ⊢ Eq (v (HSub.hSub 1 x)) 1
  -/
  rw [sub_eq_add_neg 1 x]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    x : R
    h : LT.lt (v (Neg.neg x)) (v 1)
    ⊢ Eq (v (HAdd.hAdd 1 (Neg.neg x))) 1
  -/
  simpa only [v.map_one, v.map_neg] using v.map_add_eq_of_lt_left h
  /-
    🎉 no goals
  -/


theorem one_lt_val_iff (v : Valuation K Γ₀) {x : K} (h : x ≠ 0) : 1 < v x ↔ v x⁻¹ < 1 := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    h : Ne x 0
    ⊢ Iff (LT.lt 1 (v x)) (LT.lt (v (Inv.inv x)) 1)
  -/
  simp [inv_lt_one₀ (v.pos_iff.2 h)]
  /-
    🎉 no goals
  -/


theorem one_le_val_iff (v : Valuation K Γ₀) {x : K} (h : x ≠ 0) : 1 ≤ v x ↔ v x⁻¹ ≤ 1 := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    h : Ne x 0
    ⊢ Iff (LE.le 1 (v x)) (LE.le (v (Inv.inv x)) 1)
  -/
  simp [inv_le_one₀ (v.pos_iff.2 h)]
  /-
    🎉 no goals
  -/


theorem val_lt_one_iff (v : Valuation K Γ₀) {x : K} (h : x ≠ 0) : v x < 1 ↔ 1 < v x⁻¹ := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    h : Ne x 0
    ⊢ Iff (LT.lt (v x) 1) (LT.lt 1 (v (Inv.inv x)))
  -/
  simp [one_lt_inv₀ (v.pos_iff.2 h)]
  /-
    🎉 no goals
  -/


theorem val_le_one_iff (v : Valuation K Γ₀) {x : K} (h : x ≠ 0) : v x ≤ 1 ↔ 1 ≤ v x⁻¹ := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    h : Ne x 0
    ⊢ Iff (LE.le (v x) 1) (LE.le 1 (v (Inv.inv x)))
  -/
  simp [one_le_inv₀ (v.pos_iff.2 h)]
  /-
    🎉 no goals
  -/


theorem val_eq_one_iff (v : Valuation K Γ₀) {x : K} : v x = 1 ↔ v x⁻¹ = 1 := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    ⊢ Iff (Eq (v x) 1) (Eq (v (Inv.inv x)) 1)
  -/
  by_cases h : x = 0
    /-
      case pos
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Eq x 0
      ⊢ Iff (Eq (v x) 1) (Eq (v (Inv.inv x)) 1)
    -/
  · simp only [map_inv₀, inv_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Not (Eq x 0)
      ⊢ Iff (Eq (v x) 1) (Eq (v (Inv.inv x)) 1)
    -/
  · simpa only [le_antisymm_iff, And.comm] using and_congr (one_le_val_iff v h) (val_le_one_iff v h)
    /-
      🎉 no goals
    -/


theorem val_le_one_or_val_inv_lt_one (v : Valuation K Γ₀) (x : K) : v x ≤ 1 ∨ v x⁻¹ < 1 := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    ⊢ Or (LE.le (v x) 1) (LT.lt (v (Inv.inv x)) 1)
  -/
  by_cases h : x = 0
    /-
      case pos
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Eq x 0
      ⊢ Or (LE.le (v x) 1) (LT.lt (v (Inv.inv x)) 1)
    -/
  · simp only [h, _root_.map_zero, zero_le', inv_zero, zero_lt_one, or_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Not (Eq x 0)
      ⊢ Or (LE.le (v x) 1) (LT.lt (v (Inv.inv x)) 1)
    -/
  · simp only [← one_lt_val_iff v h, le_or_lt]
    /-
      🎉 no goals
    -/


/--
This theorem is a weaker version of `Valuation.val_le_one_or_val_inv_lt_one`, but more symmetric
in `x` and `x⁻¹`.
-/
theorem val_le_one_or_val_inv_le_one (v : Valuation K Γ₀) (x : K) : v x ≤ 1 ∨ v x⁻¹ ≤ 1 := by
  /-
    K : Type u_1
    inst✝¹ : DivisionRing K
    Γ₀ : Type u_4
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation K Γ₀
    x : K
    ⊢ Or (LE.le (v x) 1) (LE.le (v (Inv.inv x)) 1)
  -/
  by_cases h : x = 0
    /-
      case pos
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Eq x 0
      ⊢ Or (LE.le (v x) 1) (LE.le (v (Inv.inv x)) 1)
    -/
  · simp only [h, _root_.map_zero, zero_le', inv_zero, or_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝¹ : DivisionRing K
      Γ₀ : Type u_4
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation K Γ₀
      x : K
      h : Not (Eq x 0)
      ⊢ Or (LE.le (v x) 1) (LE.le (v (Inv.inv x)) 1)
    -/
  · simp only [← one_le_val_iff v h, le_total]
    /-
      🎉 no goals
    -/


/-- The subgroup of elements whose valuation is less than a certain unit. -/
def ltAddSubgroup (v : Valuation R Γ₀) (γ : Γ₀ˣ) : AddSubgroup R where
  carrier := { x | v x < γ }
                  /-
                    K : Type u_1
                    F : Type u_2
                    R : Type u_3
                    inst✝³ : DivisionRing K
                    Γ₀ : Type u_4
                    Γ'₀ : Type u_5
                    Γ''₀ : Type u_6
                    inst✝² : LinearOrderedCommMonoidWithZero Γ''₀
                    inst✝¹ : Ring R
                    inst✝ : LinearOrderedCommGroupWithZero Γ₀
                    v✝ : Valuation R Γ₀
                    x y : R
                    v : Valuation R Γ₀
                    γ : Units Γ₀
                    ⊢ Membership.mem { carrier := setOf fun x => LT.lt (v x) ↑γ, add_mem' := ⋯ }.c …
                  -/
  zero_mem' := by simp
                  /-
                    🎉 no goals
                  -/
  add_mem' {x y} x_in y_in := lt_of_le_of_lt (v.map_add x y) (max_lt x_in y_in)
                      /-
                        K : Type u_1
                        F : Type u_2
                        R : Type u_3
                        inst✝³ : DivisionRing K
                        Γ₀ : Type u_4
                        Γ'₀ : Type u_5
                        Γ''₀ : Type u_6
                        inst✝² : LinearOrderedCommMonoidWithZero Γ''₀
                        inst✝¹ : Ring R
                        inst✝ : LinearOrderedCommGroupWithZero Γ₀
                        v✝ : Valuation R Γ₀
                        x y : R
                        v : Valuation R Γ₀
                        γ : Units Γ₀
                        x✝ : R
                        x_in : Membership.mem { carrier := setOf fun x => LT.lt (v x) ↑γ, add_mem' :=  …
                        ⊢ Membership.mem { carrier := setOf fun x => LT.lt (v x) ↑γ, add_mem' := ⋯, ze …
                      -/
  neg_mem' x_in := by rwa [Set.mem_setOf, map_neg]
                      /-
                        🎉 no goals
                      -/


@[refl]
theorem refl : v.IsEquiv v := fun _ _ => Iff.refl _


@[symm]
theorem symm (h : v₁.IsEquiv v₂) : v₂.IsEquiv v₁ := fun _ _ => Iff.symm (h _ _)


@[trans]
theorem trans (h₁₂ : v₁.IsEquiv v₂) (h₂₃ : v₂.IsEquiv v₃) : v₁.IsEquiv v₃ := fun _ _ =>
  Iff.trans (h₁₂ _ _) (h₂₃ _ _)


                                                                      /-
                                                                        R : Type u_3
                                                                        Γ₀ : Type u_4
                                                                        inst✝¹ : Ring R
                                                                        inst✝ : LinearOrderedCommMonoidWithZero Γ₀
                                                                        v v' : Valuation R Γ₀
                                                                        h : Eq v v'
                                                                        ⊢ v.IsEquiv v'
                                                                      -/
theorem of_eq {v' : Valuation R Γ₀} (h : v = v') : v.IsEquiv v' := by subst h; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem map {v' : Valuation R Γ₀} (f : Γ₀ →*₀ Γ'₀) (hf : Monotone f) (inf : Injective f)
    (h : v.IsEquiv v') : (v.map f hf).IsEquiv (v'.map f hf) :=
  let H : StrictMono f := hf.strictMono_of_injective inf
  fun r s =>
  calc
                                        /-
                                          R : Type u_3
                                          Γ₀ : Type u_4
                                          Γ'₀ : Type u_5
                                          inst✝² : Ring R
                                          inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
                                          inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
                                          v v' : Valuation R Γ₀
                                          f : MonoidWithZeroHom Γ₀ Γ'₀
                                          hf : Monotone ⇑f
                                          inf : Function.Injective ⇑f
                                          h : v.IsEquiv v'
                                          H : StrictMono ⇑f := Monotone.strictMono_of_injective hf inf
                                          r s : R
                                          ⊢ Iff (LE.le (f (v r)) (f (v s))) (LE.le (v r) (v s))
                                        -/
    f (v r) ≤ f (v s) ↔ v r ≤ v s := by rw [H.le_iff_le]
                                        /-
                                          🎉 no goals
                                        -/
    _ ↔ v' r ≤ v' s := h r s
                                  /-
                                    R : Type u_3
                                    Γ₀ : Type u_4
                                    Γ'₀ : Type u_5
                                    inst✝² : Ring R
                                    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
                                    inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
                                    v v' : Valuation R Γ₀
                                    f : MonoidWithZeroHom Γ₀ Γ'₀
                                    hf : Monotone ⇑f
                                    inf : Function.Injective ⇑f
                                    h : v.IsEquiv v'
                                    H : StrictMono ⇑f := Monotone.strictMono_of_injective hf inf
                                    r s : R
                                    ⊢ Iff (LE.le (v' r) (v' s)) (LE.le (f (v' r)) (f (v' s)))
                                  -/
    _ ↔ f (v' r) ≤ f (v' s) := by rw [H.le_iff_le]
                                  /-
                                    🎉 no goals
                                  -/


/-- `comap` preserves equivalence. -/
theorem comap {S : Type*} [Ring S] (f : S →+* R) (h : v₁.IsEquiv v₂) :
    (v₁.comap f).IsEquiv (v₂.comap f) := fun r s => h (f r) (f s)


theorem val_eq (h : v₁.IsEquiv v₂) {r s : R} : v₁ r = v₁ s ↔ v₂ r = v₂ s := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝² : Ring R
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
    v₁ : Valuation R Γ₀
    v₂ : Valuation R Γ'₀
    h : v₁.IsEquiv v₂
    r s : R
    ⊢ Iff (Eq (v₁ r) (v₁ s)) (Eq (v₂ r) (v₂ s))
  -/
  simpa only [le_antisymm_iff] using and_congr (h r s) (h s r)
  /-
    🎉 no goals
  -/


theorem ne_zero (h : v₁.IsEquiv v₂) {r : R} : v₁ r ≠ 0 ↔ v₂ r ≠ 0 := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝² : Ring R
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
    v₁ : Valuation R Γ₀
    v₂ : Valuation R Γ'₀
    h : v₁.IsEquiv v₂
    r : R
    ⊢ Iff (Ne (v₁ r) 0) (Ne (v₂ r) 0)
  -/
  have : v₁ r ≠ v₁ 0 ↔ v₂ r ≠ v₂ 0 := not_congr h.val_eq
  /-
    R : Type u_3
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝² : Ring R
    inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
    inst✝ : LinearOrderedCommMonoidWithZero Γ'₀
    v₁ : Valuation R Γ₀
    v₂ : Valuation R Γ'₀
    h : v₁.IsEquiv v₂
    r : R
    this : Iff (Ne (v₁ r) (v₁ 0)) (Ne (v₂ r) (v₂ 0))
    ⊢ Iff (Ne (v₁ r) 0) (Ne (v₂ r) 0)
  -/
  rwa [v₁.map_zero, v₂.map_zero] at this
  /-
    🎉 no goals
  -/


theorem isEquiv_of_map_strictMono [LinearOrderedCommMonoidWithZero Γ₀]
    [LinearOrderedCommMonoidWithZero Γ'₀] [Ring R] {v : Valuation R Γ₀} (f : Γ₀ →*₀ Γ'₀)
    (H : StrictMono f) : IsEquiv (v.map f H.monotone) v := fun _x _y =>
  ⟨H.le_iff_le.mp, fun h => H.monotone h⟩


theorem isEquiv_iff_val_lt_val [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀} :
    v.IsEquiv v' ↔ ∀ {x y : K}, v x < v y ↔ v' x < v' y := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt (v' x) (v' y …
  -/
  simp only [IsEquiv, le_iff_le_iff_lt_iff_lt]
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (∀ (r s : K), Iff (LT.lt (v s) (v r)) (LT.lt (v' s) (v' r))) (∀ {x y : K …
  -/
  exact forall_comm
  /-
    🎉 no goals
  -/


alias ⟨IsEquiv.lt_iff_lt, _⟩ := isEquiv_iff_val_lt_val


theorem isEquiv_of_val_le_one [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀}
    (h : ∀ {x : K}, v x ≤ 1 ↔ v' x ≤ 1) : v.IsEquiv v' := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    h : ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
    ⊢ v.IsEquiv v'
  -/
  intro x y
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    h : ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
    x y : K
    ⊢ Iff (LE.le (v x) (v y)) (LE.le (v' x) (v' y))
  -/
  obtain rfl | hy := eq_or_ne y 0
    /-
      case inl
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
      x : K
      ⊢ Iff (LE.le (v x) (v 0)) (LE.le (v' x) (v' 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
      x y : K
      hy : Ne y 0
      ⊢ Iff (LE.le (v x) (v y)) (LE.le (v' x) (v' y))
    -/
  · rw [← div_le_one₀, ← v.map_div, h, v'.map_div, div_le_one₀] <;>
      /-
        case inr
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
        x y : K
        hy : Ne y 0
        ⊢ LT.lt 0 (v' y)
      -/
      /-
        🎉 no goals
      -/
      rwa [zero_lt_iff, ne_zero_iff]
      /-
        🎉 no goals
      -/


theorem isEquiv_iff_val_le_one [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀} :
    v.IsEquiv v' ↔ ∀ {x : K}, v x ≤ 1 ↔ v' x ≤ 1 :=
                 /-
                   K : Type u_1
                   inst✝² : DivisionRing K
                   Γ₀ : Type u_4
                   Γ'₀ : Type u_5
                   inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
                   inst✝ : LinearOrderedCommGroupWithZero Γ'₀
                   v : Valuation K Γ₀
                   v' : Valuation K Γ'₀
                   h : v.IsEquiv v'
                   x : K
                   ⊢ Iff (LE.le (v x) 1) (LE.le (v' x) 1)
                 -/
  ⟨fun h x => by simpa using h x 1, isEquiv_of_val_le_one⟩
                 /-
                   🎉 no goals
                 -/


alias ⟨IsEquiv.le_one_iff_le_one, _⟩ := isEquiv_iff_val_le_one


theorem isEquiv_iff_val_eq_one [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀} :
    v.IsEquiv v' ↔ ∀ {x : K}, v x = 1 ↔ v' x = 1 := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (v.IsEquiv v') (∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1))
  -/
  constructor
    /-
      case mp
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      ⊢ v.IsEquiv v' → ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
    -/
  · intro h x
    /-
      case mp
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : v.IsEquiv v'
      x : K
      ⊢ Iff (Eq (v x) 1) (Eq (v' x) 1)
    -/
    simpa using @IsEquiv.val_eq _ _ _ _ _ _ v v' h x 1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      ⊢ (∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)) → v.IsEquiv v'
    -/
  · intro h
    /-
      case mpr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
      ⊢ v.IsEquiv v'
    -/
    apply isEquiv_of_val_le_one
    /-
      case mpr.h
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
      ⊢ ∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x) 1)
    -/
    intro x
    /-
      case mpr.h
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
      x : K
      ⊢ Iff (LE.le (v x) 1) (LE.le (v' x) 1)
    -/
    constructor
      /-
        case mpr.h.mp
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
        x : K
        ⊢ LE.le (v x) 1 → LE.le (v' x) 1
      -/
    · intro hx
      /-
        case mpr.h.mp
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
        x : K
        hx : LE.le (v x) 1
        ⊢ LE.le (v' x) 1
      -/
      rcases lt_or_eq_of_le hx with hx' | hx'
      · have : v (1 + x) = 1 := by
          rw [← v.map_one]
          apply map_add_eq_of_lt_left
          simpa
        /-
          case mpr.h.mp.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : LT.lt (v x) 1
          this : Eq (v (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v' x) 1
        -/
        rw [h] at this
        /-
          case mpr.h.mp.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : LT.lt (v x) 1
          this : Eq (v' (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v' x) 1
        -/
        rw [show x = -1 + (1 + x) by simp]
        /-
          case mpr.h.mp.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : LT.lt (v x) 1
          this : Eq (v' (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v' (HAdd.hAdd (-1) (HAdd.hAdd 1 x))) 1
        -/
        refine le_trans (v'.map_add _ _) ?_
        /-
          case mpr.h.mp.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : LT.lt (v x) 1
          this : Eq (v' (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (Max.max (v' (-1)) (v' (HAdd.hAdd 1 x))) 1
        -/
        simp [this]
        /-
          🎉 no goals
        -/
        /-
          case mpr.h.mp.inr
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : Eq (v x) 1
          ⊢ LE.le (v' x) 1
        -/
      · rw [h] at hx'
        /-
          case mpr.h.mp.inr
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v x) 1
          hx' : Eq (v' x) 1
          ⊢ LE.le (v' x) 1
        -/
        exact le_of_eq hx'
        /-
          🎉 no goals
        -/
      /-
        case mpr.h.mpr
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
        x : K
        ⊢ LE.le (v' x) 1 → LE.le (v x) 1
      -/
    · intro hx
      /-
        case mpr.h.mpr
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
        x : K
        hx : LE.le (v' x) 1
        ⊢ LE.le (v x) 1
      -/
      rcases lt_or_eq_of_le hx with hx' | hx'
      · have : v' (1 + x) = 1 := by
          rw [← v'.map_one]
          apply map_add_eq_of_lt_left
          simpa
        /-
          case mpr.h.mpr.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : LT.lt (v' x) 1
          this : Eq (v' (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v x) 1
        -/
        rw [← h] at this
        /-
          case mpr.h.mpr.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : LT.lt (v' x) 1
          this : Eq (v (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v x) 1
        -/
        rw [show x = -1 + (1 + x) by simp]
        /-
          case mpr.h.mpr.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : LT.lt (v' x) 1
          this : Eq (v (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (v (HAdd.hAdd (-1) (HAdd.hAdd 1 x))) 1
        -/
        refine le_trans (v.map_add _ _) ?_
        /-
          case mpr.h.mpr.inl
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : LT.lt (v' x) 1
          this : Eq (v (HAdd.hAdd 1 x)) 1
          ⊢ LE.le (Max.max (v (-1)) (v (HAdd.hAdd 1 x))) 1
        -/
        simp [this]
        /-
          🎉 no goals
        -/
        /-
          case mpr.h.mpr.inr
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : Eq (v' x) 1
          ⊢ LE.le (v x) 1
        -/
      · rw [← h] at hx'
        /-
          case mpr.h.mpr.inr
          K : Type u_1
          inst✝² : DivisionRing K
          Γ₀ : Type u_4
          Γ'₀ : Type u_5
          inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
          inst✝ : LinearOrderedCommGroupWithZero Γ'₀
          v : Valuation K Γ₀
          v' : Valuation K Γ'₀
          h : ∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1)
          x : K
          hx : LE.le (v' x) 1
          hx' : Eq (v x) 1
          ⊢ LE.le (v x) 1
        -/
        exact le_of_eq hx'
        /-
          🎉 no goals
        -/


alias ⟨IsEquiv.eq_one_iff_eq_one, _⟩ := isEquiv_iff_val_eq_one


theorem isEquiv_iff_val_lt_one [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀} :
    v.IsEquiv v' ↔ ∀ {x : K}, v x < 1 ↔ v' x < 1 := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (v.IsEquiv v') (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1))
  -/
  constructor
    /-
      case mp
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      ⊢ v.IsEquiv v' → ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
    -/
  · intro h x
    simp only [lt_iff_le_and_ne,
      and_congr h.le_one_iff_le_one h.eq_one_iff_eq_one.not]
    /-
      case mpr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      ⊢ (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)) → v.IsEquiv v'
    -/
  · rw [isEquiv_iff_val_eq_one]
    /-
      case mpr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      ⊢ (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)) → ∀ {x : K}, Iff (Eq (v x) …
    -/
    intro h x
    /-
      case mpr
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
      x : K
      ⊢ Iff (Eq (v x) 1) (Eq (v' x) 1)
    -/
    by_cases hx : x = 0
      /-
        case pos
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
        x : K
        hx : Eq x 0
        ⊢ Iff (Eq (v x) 1) (Eq (v' x) 1)
      -/
    · simp only [(zero_iff _).2 hx, zero_ne_one]
      /-
        🎉 no goals
      -/
    /-
      case neg
      K : Type u_1
      inst✝² : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
      inst✝ : LinearOrderedCommGroupWithZero Γ'₀
      v : Valuation K Γ₀
      v' : Valuation K Γ'₀
      h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
      x : K
      hx : Not (Eq x 0)
      ⊢ Iff (Eq (v x) 1) (Eq (v' x) 1)
    -/
    constructor
      /-
        case neg.mp
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
        x : K
        hx : Not (Eq x 0)
        ⊢ Eq (v x) 1 → Eq (v' x) 1
      -/
    · intro hh
      /-
        case neg.mp
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
        x : K
        hx : Not (Eq x 0)
        hh : Eq (v x) 1
        ⊢ Eq (v' x) 1
      -/
      by_contra h_1
      cases ne_iff_lt_or_gt.1 h_1 with
      | inl h_2 => simpa [hh, lt_self_iff_false] using h.2 h_2
      | inr h_2 =>
          rw [← inv_one, ← inv_eq_iff_eq_inv, ← map_inv₀] at hh
          exact hh.not_lt (h.2 ((one_lt_val_iff v' hx).1 h_2))
      /-
        case neg.mpr
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
        x : K
        hx : Not (Eq x 0)
        ⊢ Eq (v' x) 1 → Eq (v x) 1
      -/
    · intro hh
      /-
        case neg.mpr
        K : Type u_1
        inst✝² : DivisionRing K
        Γ₀ : Type u_4
        Γ'₀ : Type u_5
        inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
        inst✝ : LinearOrderedCommGroupWithZero Γ'₀
        v : Valuation K Γ₀
        v' : Valuation K Γ'₀
        h : ∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)
        x : K
        hx : Not (Eq x 0)
        hh : Eq (v' x) 1
        ⊢ Eq (v x) 1
      -/
      by_contra h_1
      cases ne_iff_lt_or_gt.1 h_1 with
      | inl h_2 => simpa [hh, lt_self_iff_false] using h.1 h_2
      | inr h_2 =>
        rw [← inv_one, ← inv_eq_iff_eq_inv, ← map_inv₀] at hh
        exact hh.not_lt (h.1 ((one_lt_val_iff v hx).1 h_2))


alias ⟨IsEquiv.lt_one_iff_lt_one, _⟩ := isEquiv_iff_val_lt_one


theorem isEquiv_iff_val_sub_one_lt_one [LinearOrderedCommGroupWithZero Γ₀]
    [LinearOrderedCommGroupWithZero Γ'₀] {v : Valuation K Γ₀} {v' : Valuation K Γ'₀} :
    v.IsEquiv v' ↔ ∀ {x : K}, v (x - 1) < 1 ↔ v' (x - 1) < 1 := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (v.IsEquiv v') (∀ {x : K}, Iff (LT.lt (v (HSub.hSub x 1)) 1) (LT.lt (v'  …
  -/
  rw [isEquiv_iff_val_lt_one]
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ Iff (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x) 1)) (∀ {x : K}, Iff (LT.lt …
  -/
  exact (Equiv.subRight 1).surjective.forall
  /-
    🎉 no goals
  -/


alias ⟨IsEquiv.val_sub_one_lt_one_iff, _⟩ := isEquiv_iff_val_sub_one_lt_one


theorem isEquiv_tfae [LinearOrderedCommGroupWithZero Γ₀] [LinearOrderedCommGroupWithZero Γ'₀]
    (v : Valuation K Γ₀) (v' : Valuation K Γ'₀) :
    [ v.IsEquiv v',
      ∀ {x y}, v x < v y ↔ v' x < v' y,
      ∀ {x}, v x ≤ 1 ↔ v' x ≤ 1,
      ∀ {x}, v x = 1 ↔ v' x = 1,
      ∀ {x}, v x < 1 ↔ v' x < 1,
      ∀ {x}, v (x - 1) < 1 ↔ v' (x - 1) < 1 ].TFAE := by
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_have 1 ↔ 2 := isEquiv_iff_val_lt_val
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    tfae_1_iff_2 : Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt …
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_have 1 ↔ 3 := isEquiv_iff_val_le_one
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    tfae_1_iff_2 : Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt …
    tfae_1_iff_3 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x …
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_have 1 ↔ 4 := isEquiv_iff_val_eq_one
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    tfae_1_iff_2 : Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt …
    tfae_1_iff_3 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x …
    tfae_1_iff_4 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1))
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_have 1 ↔ 5 := isEquiv_iff_val_lt_one
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    tfae_1_iff_2 : Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt …
    tfae_1_iff_3 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x …
    tfae_1_iff_4 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1))
    tfae_1_iff_5 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x …
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_have 1 ↔ 6 := isEquiv_iff_val_sub_one_lt_one
  /-
    K : Type u_1
    inst✝² : DivisionRing K
    Γ₀ : Type u_4
    Γ'₀ : Type u_5
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    inst✝ : LinearOrderedCommGroupWithZero Γ'₀
    v : Valuation K Γ₀
    v' : Valuation K Γ'₀
    tfae_1_iff_2 : Iff (v.IsEquiv v') (∀ {x y : K}, Iff (LT.lt (v x) (v y)) (LT.lt …
    tfae_1_iff_3 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LE.le (v x) 1) (LE.le (v' x …
    tfae_1_iff_4 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (Eq (v x) 1) (Eq (v' x) 1))
    tfae_1_iff_5 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LT.lt (v x) 1) (LT.lt (v' x …
    tfae_1_iff_6 : Iff (v.IsEquiv v') (∀ {x : K}, Iff (LT.lt (v (HSub.hSub x 1)) 1 …
    ⊢ (List.cons (v.IsEquiv v') (List.cons (∀ {x y : K}, Iff (LT.lt (v x) (v y)) ( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- The support of a valuation `v : R → Γ₀` is the ideal of `R` where `v` vanishes. -/
def supp : Ideal R where
  carrier := { x | v x = 0 }
  zero_mem' := map_zero v
  add_mem' {x y} hx hy := le_zero_iff.mp <|
    calc
      v (x + y) ≤ max (v x) (v y) := v.map_add x y
      _ ≤ 0 := max_le (le_zero_iff.mpr hx) (le_zero_iff.mpr hy)
  smul_mem' c x hx :=
    calc
      v (c * x) = v c * v x := map_mul v c x
      _ = v c * 0 := congr_arg _ hx
      _ = 0 := mul_zero _


@[simp]
theorem mem_supp_iff (x : R) : x ∈ supp v ↔ v x = 0 :=
  Iff.rfl


/-- The support of a valuation is a prime ideal. -/
instance [Nontrivial Γ₀] [NoZeroDivisors Γ₀] : Ideal.IsPrime (supp v) :=
  ⟨fun h =>
    one_ne_zero (α := Γ₀) <|
      calc
        1 = v 1 := v.map_one.symm
                    /-
                      K : Type u_1
                      F : Type u_2
                      R : Type u_3
                      inst✝⁵ : DivisionRing K
                      Γ₀ : Type u_4
                      Γ'₀ : Type u_5
                      Γ''₀ : Type u_6
                      inst✝⁴ : LinearOrderedCommMonoidWithZero Γ''₀
                      inst✝³ : CommRing R
                      inst✝² : LinearOrderedCommMonoidWithZero Γ₀
                      v : Valuation R Γ₀
                      inst✝¹ : Nontrivial Γ₀
                      inst✝ : NoZeroDivisors Γ₀
                      h : Eq v.supp Top.top
                      ⊢ Eq (v 1) 0
                    -/
        _ = 0 := by rw [← mem_supp_iff, h]; exact Submodule.mem_top,
                                            /-
                                              🎉 no goals
                                            -/
   fun {x y} hxy => by
    /-
      K : Type u_1
      F : Type u_2
      R : Type u_3
      inst✝⁵ : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      Γ''₀ : Type u_6
      inst✝⁴ : LinearOrderedCommMonoidWithZero Γ''₀
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      inst✝¹ : Nontrivial Γ₀
      inst✝ : NoZeroDivisors Γ₀
      x y : R
      hxy : Membership.mem v.supp (HMul.hMul x y)
      ⊢ Or (Membership.mem v.supp x) (Membership.mem v.supp y)
    -/
    simp only [mem_supp_iff] at hxy ⊢
    /-
      K : Type u_1
      F : Type u_2
      R : Type u_3
      inst✝⁵ : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      Γ''₀ : Type u_6
      inst✝⁴ : LinearOrderedCommMonoidWithZero Γ''₀
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      inst✝¹ : Nontrivial Γ₀
      inst✝ : NoZeroDivisors Γ₀
      x y : R
      hxy : Eq (v (HMul.hMul x y)) 0
      ⊢ Or (Eq (v x) 0) (Eq (v y) 0)
    -/
    rw [v.map_mul x y] at hxy
    /-
      K : Type u_1
      F : Type u_2
      R : Type u_3
      inst✝⁵ : DivisionRing K
      Γ₀ : Type u_4
      Γ'₀ : Type u_5
      Γ''₀ : Type u_6
      inst✝⁴ : LinearOrderedCommMonoidWithZero Γ''₀
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      inst✝¹ : Nontrivial Γ₀
      inst✝ : NoZeroDivisors Γ₀
      x y : R
      hxy : Eq (HMul.hMul (v x) (v y)) 0
      ⊢ Or (Eq (v x) 0) (Eq (v y) 0)
    -/
    exact eq_zero_or_eq_zero_of_mul_eq_zero hxy⟩
    /-
      🎉 no goals
    -/


theorem map_add_supp (a : R) {s : R} (h : s ∈ supp v) : v (a + s) = v a := by
  have aux : ∀ a s, v s = 0 → v (a + s) ≤ v a := by
    intro a' s' h'
    refine le_trans (v.map_add a' s') (max_le le_rfl ?_)
    simp [h']
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    a s : R
    h : Membership.mem v.supp s
    aux : ∀ (a s : R), Eq (v s) 0 → LE.le (v (HAdd.hAdd a s)) (v a)
    ⊢ Eq (v (HAdd.hAdd a s)) (v a)
  -/
  apply le_antisymm (aux a s h)
  calc
    v a = v (a + s + -s) := by simp
    _ ≤ v (a + s) := aux (a + s) (-s) (by rwa [← Ideal.neg_mem_iff] at h)


theorem comap_supp {S : Type*} [CommRing S] (f : S →+* R) :
    supp (v.comap f) = Ideal.comap f v.supp :=
                        /-
                          R : Type u_3
                          Γ₀ : Type u_4
                          inst✝² : CommRing R
                          inst✝¹ : LinearOrderedCommMonoidWithZero Γ₀
                          v : Valuation R Γ₀
                          S : Type u_7
                          inst✝ : CommRing S
                          f : RingHom S R
                          x : S
                          ⊢ Iff (Membership.mem (Valuation.comap f v).supp x) (Membership.mem (Ideal.com …
                        -/
  Ideal.ext fun x => by rw [mem_supp_iff, Ideal.mem_comap, mem_supp_iff, comap_apply]
                        /-
                          🎉 no goals
                        -/


/-- The type of `Γ₀`-valued additive valuations on `R`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
def AddValuation :=
  Valuation R (Multiplicative Γ₀ᵒᵈ)


/-- A valuation is coerced to the underlying function `R → Γ₀`. -/
instance (R) (Γ₀) [Ring R] [LinearOrderedAddCommMonoidWithTop Γ₀] :
    FunLike (AddValuation R Γ₀) R Γ₀ where
  coe v := v.toMonoidWithZeroHom.toFun
                           /-
                             K : Type u_1
                             F : Type u_2
                             R✝ : Type u_3
                             inst✝² : DivisionRing K
                             Γ₀✝ : Type u_4
                             Γ'₀ : Type u_5
                             R : Type ?u.157456
                             Γ₀ : Type ?u.157459
                             inst✝¹ : Ring R
                             inst✝ : LinearOrderedAddCommMonoidWithTop Γ₀
                             f g : AddValuation R Γ₀
                             ⊢ Eq ((fun v => (↑v.toMonoidWithZeroHom).toFun) f) ((fun v => (↑v.toMonoidWith …
                           -/
  coe_injective' f g := by cases f; cases g; simp (config := {contextual := true})
                                             /-
                                               🎉 no goals
                                             -/


/-- An alternate constructor of `AddValuation`, that doesn't reference `Multiplicative Γ₀ᵒᵈ` -/
def of : AddValuation R Γ₀ where
  toFun := f
  map_one' := h1
  map_zero' := h0
  map_add_le_max' := hadd
  map_mul' := hmul


@[simp]
theorem of_apply : (of f h0 h1 hadd hmul) r = f r := rfl


/-- The `Valuation` associated to an `AddValuation` (useful if the latter is constructed using
`AddValuation.of`). -/
def toValuation : AddValuation R Γ₀ ≃ Valuation R (Multiplicative Γ₀ᵒᵈ) :=
  Equiv.refl _


@[deprecated (since := "2024-11-09")]
alias valuation := toValuation


/-- The `AddValuation` associated to a `Valuation`.
-/
def ofValuation : Valuation R (Multiplicative Γ₀ᵒᵈ) ≃ AddValuation R Γ₀ :=
  Equiv.refl _


@[simp]
lemma ofValuation_symm_eq : ofValuation.symm = toValuation (R := R) (Γ₀ := Γ₀) := rfl


@[simp]
lemma toValuation_symm_eq : toValuation.symm = ofValuation (R := R) (Γ₀ := Γ₀) := rfl


@[simp]
lemma ofValuation_toValuation : ofValuation (toValuation v) = v := rfl


@[simp]
lemma toValuation_ofValuation (v : Valuation R (Multiplicative Γ₀ᵒᵈ)) :
    toValuation (ofValuation v) = v := rfl


@[simp]
theorem toValuation_apply (r : R) :
    toValuation v r = Multiplicative.ofAdd (OrderDual.toDual (v r)) :=
  rfl


@[deprecated (since := "2024-11-09")]
alias valuation_apply := toValuation_apply


@[simp]
theorem ofValuation_apply (v : Valuation R (Multiplicative Γ₀ᵒᵈ)) (r : R) :
    ofValuation v r = OrderDual.ofDual (Multiplicative.toAdd (v r)) :=
  rfl


@[simp]
theorem map_zero : v 0 = (⊤ : Γ₀) :=
  Valuation.map_zero v


@[simp]
theorem map_one : v 1 = (0 : Γ₀) :=
  Valuation.map_one v

/- Porting note: helper wrapper to coerce `v` to the correct function type -/

/-- A helper function for Lean to inferring types correctly -/
def asFun : R → Γ₀ := v


@[simp]
theorem map_mul : ∀ (x y : R), v (x * y) = v x + v y :=
  Valuation.map_mul v

-- Porting note: LHS simplified so created map_add' and removed simp tag

theorem map_add : ∀ (x y : R), min (v x) (v y) ≤ v (x + y) :=
  Valuation.map_add v


@[simp]
theorem map_add' : ∀ (x y : R), v x ≤ v (x + y) ∨ v y ≤ v (x + y) := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedAddCommMonoidWithTop Γ₀
    v : AddValuation R Γ₀
    ⊢ ∀ (x y : R), Or (LE.le (v x) (v (HAdd.hAdd x y))) (LE.le (v y) (v (HAdd.hAdd …
  -/
  intro x y
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedAddCommMonoidWithTop Γ₀
    v : AddValuation R Γ₀
    x y : R
    ⊢ Or (LE.le (v x) (v (HAdd.hAdd x y))) (LE.le (v y) (v (HAdd.hAdd x y)))
  -/
  rw [← @min_le_iff _ _ (v x) (v y) (v (x+y)), ← ge_iff_le]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : Ring R
    inst✝ : LinearOrderedAddCommMonoidWithTop Γ₀
    v : AddValuation R Γ₀
    x y : R
    ⊢ GE.ge (v (HAdd.hAdd x y)) (Min.min (v x) (v y))
  -/
  apply map_add
  /-
    🎉 no goals
  -/


theorem map_le_add {x y : R} {g : Γ₀} (hx : g ≤ v x) (hy : g ≤ v y) : g ≤ v (x + y) :=
  Valuation.map_add_le v hx hy


theorem map_lt_add {x y : R} {g : Γ₀} (hx : g < v x) (hy : g < v y) : g < v (x + y) :=
  Valuation.map_add_lt v hx hy


theorem map_le_sum {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hf : ∀ i ∈ s, g ≤ v (f i)) :
    g ≤ v (∑ i ∈ s, f i) :=
  v.map_sum_le hf


theorem map_lt_sum {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hg : g ≠ ⊤)
    (hf : ∀ i ∈ s, g < v (f i)) : g < v (∑ i ∈ s, f i) :=
  v.map_sum_lt hg hf


theorem map_lt_sum' {ι : Type*} {s : Finset ι} {f : ι → R} {g : Γ₀} (hg : g < ⊤)
    (hf : ∀ i ∈ s, g < v (f i)) : g < v (∑ i ∈ s, f i) :=
  v.map_sum_lt' hg hf


@[simp]
theorem map_pow : ∀ (x : R) (n : ℕ), v (x ^ n) = n • (v x) :=
  Valuation.map_pow v


@[ext]
theorem ext {v₁ v₂ : AddValuation R Γ₀} (h : ∀ r, v₁ r = v₂ r) : v₁ = v₂ :=
  Valuation.ext h

-- The following definition is not an instance, because we have more than one `v` on a given `R`.
-- In addition, type class inference would not be able to infer `v`.

/-- If `v` is an additive valuation on a division ring then `v(x) = ⊤` iff `x = 0`. -/
@[simp]
theorem top_iff [Nontrivial Γ₀] (v : AddValuation K Γ₀) {x : K} : v x = (⊤ : Γ₀) ↔ x = 0 :=
  v.zero_iff


theorem ne_top_iff [Nontrivial Γ₀] (v : AddValuation K Γ₀) {x : K} : v x ≠ (⊤ : Γ₀) ↔ x ≠ 0 :=
  v.ne_zero_iff


/-- A ring homomorphism `S → R` induces a map `AddValuation R Γ₀ → AddValuation S Γ₀`. -/
def comap {S : Type*} [Ring S] (f : S →+* R) (v : AddValuation R Γ₀) : AddValuation S Γ₀ :=
  Valuation.comap f v


@[simp]
theorem comap_id : v.comap (RingHom.id R) = v :=
  Valuation.comap_id v


theorem comap_comp {S₁ : Type*} {S₂ : Type*} [Ring S₁] [Ring S₂] (f : S₁ →+* S₂) (g : S₂ →+* R) :
    v.comap (g.comp f) = (v.comap g).comap f :=
  Valuation.comap_comp v f g


/-- A `≤`-preserving, `⊤`-preserving group homomorphism `Γ₀ → Γ'₀` induces a map
  `AddValuation R Γ₀ → AddValuation R Γ'₀`.
-/
def map (f : Γ₀ →+ Γ'₀) (ht : f ⊤ = ⊤) (hf : Monotone f) (v : AddValuation R Γ₀) :
    AddValuation R Γ'₀ :=
  @Valuation.map R (Multiplicative Γ₀ᵒᵈ) (Multiplicative Γ'₀ᵒᵈ) _ _ _
    { toFun := f
      map_mul' := f.map_add
      map_one' := f.map_zero
      map_zero' := ht } (fun _ _ h => hf h) v


@[simp]
lemma map_apply (f : Γ₀ →+ Γ'₀) (ht : f ⊤ = ⊤) (hf : Monotone f) (v : AddValuation R Γ₀) (r : R) :
    v.map f ht hf r = f (v r) := rfl


/-- Two additive valuations on `R` are defined to be equivalent if they induce the same
  preorder on `R`. -/
def IsEquiv (v₁ : AddValuation R Γ₀) (v₂ : AddValuation R Γ'₀) : Prop :=
  Valuation.IsEquiv v₁ v₂


@[simp]
theorem map_inv (v : AddValuation K Γ₀) {x : K} : v x⁻¹ = - (v x) :=
  map_inv₀ (toValuation v) x


@[simp]
theorem map_div (v : AddValuation K Γ₀) {x y : K} : v (x / y) = v x - v y :=
  map_div₀ (toValuation v) x y


@[simp]
theorem map_neg (x : R) : v (-x) = v x :=
  Valuation.map_neg v x


theorem map_sub_swap (x y : R) : v (x - y) = v (y - x) :=
  Valuation.map_sub_swap v x y


theorem map_sub (x y : R) : min (v x) (v y) ≤ v (x - y) :=
  Valuation.map_sub v x y


theorem map_le_sub {x y : R} {g : Γ₀} (hx : g ≤ v x) (hy : g ≤ v y) : g ≤ v (x - y) :=
  Valuation.map_sub_le v hx hy


theorem map_add_of_distinct_val (h : v x ≠ v y) : v (x + y) = @Min.min Γ₀ _ (v x) (v y) :=
  Valuation.map_add_of_distinct_val v h


theorem map_add_eq_of_lt_left {x y : R} (h : v x < v y) :
    v (x + y) = v x := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedAddCommGroupWithTop Γ₀
    inst✝ : Ring R
    v : AddValuation R Γ₀
    x y : R
    h : LT.lt (v x) (v y)
    ⊢ Eq (v (HAdd.hAdd x y)) (v x)
  -/
  rw [map_add_of_distinct_val _ h.ne, min_eq_left h.le]
  /-
    🎉 no goals
  -/


theorem map_add_eq_of_lt_right {x y : R} (hx : v y < v x) :
    v (x + y) = v y := add_comm y x ▸ map_add_eq_of_lt_left v hx


theorem map_sub_eq_of_lt_left {x y : R} (hx : v x < v y) :
    v (x - y) = v x := by
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedAddCommGroupWithTop Γ₀
    inst✝ : Ring R
    v : AddValuation R Γ₀
    x y : R
    hx : LT.lt (v x) (v y)
    ⊢ Eq (v (HSub.hSub x y)) (v x)
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedAddCommGroupWithTop Γ₀
    inst✝ : Ring R
    v : AddValuation R Γ₀
    x y : R
    hx : LT.lt (v x) (v y)
    ⊢ Eq (v (HAdd.hAdd x (Neg.neg y))) (v x)
  -/
  apply map_add_eq_of_lt_left
  /-
    case h
    R : Type u_3
    Γ₀ : Type u_4
    inst✝¹ : LinearOrderedAddCommGroupWithTop Γ₀
    inst✝ : Ring R
    v : AddValuation R Γ₀
    x y : R
    hx : LT.lt (v x) (v y)
    ⊢ LT.lt (v x) (v (Neg.neg y))
  -/
  rwa [map_neg]
  /-
    🎉 no goals
  -/


theorem map_sub_eq_of_lt_right {x y : R} (hx : v y < v x) :
    v (x - y) = v y := map_sub_swap v x y ▸ map_sub_eq_of_lt_left v hx


theorem map_eq_of_lt_sub (h : v x < v (y - x)) : v y = v x :=
  Valuation.map_eq_of_sub_lt v h


@[refl]
theorem refl : v.IsEquiv v :=
  Valuation.IsEquiv.refl


@[symm]
theorem symm (h : v₁.IsEquiv v₂) : v₂.IsEquiv v₁ :=
  Valuation.IsEquiv.symm h


@[trans]
theorem trans (h₁₂ : v₁.IsEquiv v₂) (h₂₃ : v₂.IsEquiv v₃) : v₁.IsEquiv v₃ :=
  Valuation.IsEquiv.trans h₁₂ h₂₃


theorem of_eq {v' : AddValuation R Γ₀} (h : v = v') : v.IsEquiv v' :=
  Valuation.IsEquiv.of_eq h


theorem map {v' : AddValuation R Γ₀} (f : Γ₀ →+ Γ'₀) (ht : f ⊤ = ⊤) (hf : Monotone f)
    (inf : Injective f) (h : v.IsEquiv v') : (v.map f ht hf).IsEquiv (v'.map f ht hf) :=
  @Valuation.IsEquiv.map R (Multiplicative Γ₀ᵒᵈ) (Multiplicative Γ'₀ᵒᵈ) _ _ _ _ _
    { toFun := f
      map_mul' := f.map_add
      map_one' := f.map_zero
      map_zero' := ht } (fun _x _y h => hf h) inf h


/-- `comap` preserves equivalence. -/
theorem comap {S : Type*} [Ring S] (f : S →+* R) (h : v₁.IsEquiv v₂) :
    (v₁.comap f).IsEquiv (v₂.comap f) :=
  Valuation.IsEquiv.comap f h


theorem val_eq (h : v₁.IsEquiv v₂) {r s : R} : v₁ r = v₁ s ↔ v₂ r = v₂ s :=
  Valuation.IsEquiv.val_eq h


theorem ne_top (h : v₁.IsEquiv v₂) {r : R} : v₁ r ≠ (⊤ : Γ₀) ↔ v₂ r ≠ (⊤ : Γ'₀) :=
  Valuation.IsEquiv.ne_zero h


/-- The support of an additive valuation `v : R → Γ₀` is the ideal of `R` where `v x = ⊤` -/
def supp : Ideal R :=
  Valuation.supp v


@[simp]
theorem mem_supp_iff (x : R) : x ∈ supp v ↔ v x = (⊤ : Γ₀) :=
  Valuation.mem_supp_iff v x


theorem map_add_supp (a : R) {s : R} (h : s ∈ supp v) : v (a + s) = v a :=
  Valuation.map_add_supp v a h


/-- The `AddValuation` associated to a `Valuation`. -/
def toAddValuation : Valuation R Γ₀ ≃ AddValuation R (Additive Γ₀)ᵒᵈ :=
  AddValuation.ofValuation (R := R) (Γ₀ := (Additive Γ₀)ᵒᵈ)


/-- The `Valuation` associated to a `AddValuation`.
-/
def ofAddValuation : AddValuation R (Additive Γ₀)ᵒᵈ ≃ Valuation R Γ₀ :=
  AddValuation.toValuation


@[simp]
lemma ofAddValuation_symm_eq : ofAddValuation.symm = toAddValuation (R := R) (Γ₀ := Γ₀) := rfl


@[simp]
lemma toAddValuation_symm_eq : toAddValuation.symm = ofAddValuation (R := R) (Γ₀ := Γ₀) := rfl


@[simp]
lemma ofAddValuation_toAddValuation (v : Valuation R Γ₀) :
  ofAddValuation (toAddValuation v) = v := rfl


@[simp]
lemma toValuation_ofValuation (v : AddValuation R (Additive Γ₀)ᵒᵈ) :
    toAddValuation (ofAddValuation v) = v := rfl


@[simp]
theorem toAddValuation_apply (v : Valuation R Γ₀) (r : R) :
    toAddValuation v r = OrderDual.toDual (Additive.ofMul (v r)) :=
  rfl


@[simp]
theorem ofAddValuation_apply (v : AddValuation R (Additive Γ₀)ᵒᵈ) (r : R) :
    ofAddValuation v r = Additive.toMul (OrderDual.ofDual (v r)) :=
  rfl


