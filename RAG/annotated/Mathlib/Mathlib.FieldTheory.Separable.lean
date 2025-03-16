/-- A polynomial is separable iff it is coprime with its derivative. -/
@[stacks 09H1 "first part"]
def Separable (f : R[X]) : Prop :=
  IsCoprime f (derivative f)


theorem separable_def (f : R[X]) : f.Separable ↔ IsCoprime f (derivative f) :=
  Iff.rfl


theorem separable_def' (f : R[X]) : f.Separable ↔ ∃ a b : R[X], a * f + b * (derivative f) = 1 :=
  Iff.rfl


theorem not_separable_zero [Nontrivial R] : ¬Separable (0 : R[X]) := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Not (Polynomial.Separable 0)
  -/
  rintro ⟨x, y, h⟩
  /-
    case intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    x y : Polynomial R
    h : Eq (HAdd.hAdd (HMul.hMul x 0) (HMul.hMul y (Polynomial.derivative 0))) 1
    ⊢ False
  -/
  simp only [derivative_zero, mul_zero, add_zero, zero_ne_one] at h
  /-
    🎉 no goals
  -/


theorem Separable.ne_zero [Nontrivial R] {f : R[X]} (h : f.Separable) : f ≠ 0 :=
  (not_separable_zero <| · ▸ h)


@[simp]
theorem separable_one : (1 : R[X]).Separable :=
  isCoprime_one_left


@[nontriviality]
theorem separable_of_subsingleton [Subsingleton R] (f : R[X]) : f.Separable := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : Subsingleton R
    f : Polynomial R
    ⊢ f.Separable
  -/
  simp [Separable, IsCoprime, eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


theorem separable_X_add_C (a : R) : (X + C a).Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    a : R
    ⊢ (HAdd.hAdd Polynomial.X (Polynomial.C a)).Separable
  -/
  rw [separable_def, derivative_add, derivative_X, derivative_C, add_zero]
  /-
    R : Type u
    inst✝ : CommSemiring R
    a : R
    ⊢ IsCoprime (HAdd.hAdd Polynomial.X (Polynomial.C a)) 1
  -/
  exact isCoprime_one_right
  /-
    🎉 no goals
  -/


theorem separable_X : (X : R[X]).Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Polynomial.X.Separable
  -/
  rw [separable_def, derivative_X]
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ IsCoprime Polynomial.X 1
  -/
  exact isCoprime_one_right
  /-
    🎉 no goals
  -/


theorem separable_C (r : R) : (C r).Separable ↔ IsUnit r := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    r : R
    ⊢ Iff (Polynomial.C r).Separable (IsUnit r)
  -/
  rw [separable_def, derivative_C, isCoprime_zero_right, isUnit_C]
  /-
    🎉 no goals
  -/


theorem Separable.of_mul_left {f g : R[X]} (h : (f * g).Separable) : f.Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul f g).Separable
    ⊢ f.Separable
  -/
  have := h.of_mul_left_left; rw [derivative_mul] at this
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul f g).Separable
    this : IsCoprime f (HAdd.hAdd (HMul.hMul (Polynomial.derivative f) g) (HMul.hM …
    ⊢ f.Separable
  -/
  exact IsCoprime.of_mul_right_left (IsCoprime.of_add_mul_left_right this)
  /-
    🎉 no goals
  -/


theorem Separable.of_mul_right {f g : R[X]} (h : (f * g).Separable) : g.Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul f g).Separable
    ⊢ g.Separable
  -/
  rw [mul_comm] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul g f).Separable
    ⊢ g.Separable
  -/
  exact h.of_mul_left
  /-
    🎉 no goals
  -/


theorem Separable.of_dvd {f g : R[X]} (hf : f.Separable) (hfg : g ∣ f) : g.Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    hf : f.Separable
    hfg : Dvd.dvd g f
    ⊢ g.Separable
  -/
  rcases hfg with ⟨f', rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    g f' : Polynomial R
    hf : (HMul.hMul g f').Separable
    ⊢ g.Separable
  -/
  exact Separable.of_mul_left hf
  /-
    🎉 no goals
  -/


theorem separable_gcd_left {F : Type*} [Field F] [DecidableEq F[X]]
    {f : F[X]} (hf : f.Separable) (g : F[X]) :
    (EuclideanDomain.gcd f g).Separable :=
  Separable.of_dvd hf (EuclideanDomain.gcd_dvd_left f g)


theorem separable_gcd_right {F : Type*} [Field F] [DecidableEq F[X]]
    {g : F[X]} (f : F[X]) (hg : g.Separable) :
    (EuclideanDomain.gcd f g).Separable :=
  Separable.of_dvd hg (EuclideanDomain.gcd_dvd_right f g)


theorem Separable.isCoprime {f g : R[X]} (h : (f * g).Separable) : IsCoprime f g := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul f g).Separable
    ⊢ IsCoprime f g
  -/
  have := h.of_mul_left_left; rw [derivative_mul] at this
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : (HMul.hMul f g).Separable
    this : IsCoprime f (HAdd.hAdd (HMul.hMul (Polynomial.derivative f) g) (HMul.hM …
    ⊢ IsCoprime f g
  -/
  exact IsCoprime.of_mul_right_right (IsCoprime.of_add_mul_left_right this)
  /-
    🎉 no goals
  -/


theorem Separable.of_pow' {f : R[X]} :
    ∀ {n : ℕ} (_h : (f ^ n).Separable), IsUnit f ∨ f.Separable ∧ n = 1 ∨ n = 0
  | 0 => fun _h => Or.inr <| Or.inr rfl
  | 1 => fun h => Or.inr <| Or.inl ⟨pow_one f ▸ h, rfl⟩
  | n + 2 => fun h => by
    /-
      R : Type u
      inst✝ : CommSemiring R
      f : Polynomial R
      n : Nat
      h : (HPow.hPow f (HAdd.hAdd n 2)).Separable
      ⊢ Or (IsUnit f) (Or (And f.Separable (Eq (HAdd.hAdd n 2) 1)) (Eq (HAdd.hAdd n  …
    -/
    rw [pow_succ, pow_succ] at h
    /-
      R : Type u
      inst✝ : CommSemiring R
      f : Polynomial R
      n : Nat
      h : (HMul.hMul (HMul.hMul (HPow.hPow f n) f) f).Separable
      ⊢ Or (IsUnit f) (Or (And f.Separable (Eq (HAdd.hAdd n 2) 1)) (Eq (HAdd.hAdd n  …
    -/
    exact Or.inl (isCoprime_self.1 h.isCoprime.of_mul_left_right)
    /-
      🎉 no goals
    -/


theorem Separable.of_pow {f : R[X]} (hf : ¬IsUnit f) {n : ℕ} (hn : n ≠ 0)
    (hfs : (f ^ n).Separable) : f.Separable ∧ n = 1 :=
  (hfs.of_pow'.resolve_left hf).resolve_right hn


theorem Separable.map {p : R[X]} (h : p.Separable) {f : R →+* S} : (p.map f).Separable :=
  let ⟨a, b, H⟩ := h
  ⟨a.map f, b.map f, by
    rw [derivative_map, ← Polynomial.map_mul, ← Polynomial.map_mul, ← Polynomial.map_add, H,
      Polynomial.map_one]⟩


theorem _root_.Associated.separable {f g : R[X]}
    (ha : Associated f g) (h : f.Separable) : g.Separable := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    ha : Associated f g
    h : f.Separable
    ⊢ g.Separable
  -/
  obtain ⟨⟨u, v, h1, h2⟩, ha⟩ := ha
  /-
    case intro.mk
    R : Type u
    inst✝ : CommSemiring R
    f g : Polynomial R
    h : f.Separable
    u v : Polynomial R
    h1 : Eq (HMul.hMul u v) 1
    h2 : Eq (HMul.hMul v u) 1
    ha : Eq (HMul.hMul f ↑{ val := u, inv := v, val_inv := h1, inv_val := h2 }) g
    ⊢ g.Separable
  -/
  obtain ⟨a, b, h⟩ := h
  /-
    case intro.mk.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    f g u v : Polynomial R
    h1 : Eq (HMul.hMul u v) 1
    h2 : Eq (HMul.hMul v u) 1
    ha : Eq (HMul.hMul f ↑{ val := u, inv := v, val_inv := h1, inv_val := h2 }) g
    a b : Polynomial R
    h : Eq (HAdd.hAdd (HMul.hMul a f) (HMul.hMul b (Polynomial.derivative f))) 1
    ⊢ g.Separable
  -/
  refine ⟨a * v + b * derivative v, b * v, ?_⟩
  /-
    case intro.mk.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    f g u v : Polynomial R
    h1 : Eq (HMul.hMul u v) 1
    h2 : Eq (HMul.hMul v u) 1
    ha : Eq (HMul.hMul f ↑{ val := u, inv := v, val_inv := h1, inv_val := h2 }) g
    a b : Polynomial R
    h : Eq (HAdd.hAdd (HMul.hMul a f) (HMul.hMul b (Polynomial.derivative f))) 1
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a v) (HMul.hMul b (Polynomial …
  -/
  replace h := congr($h * $(h1))
  /-
    case intro.mk.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    f g u v : Polynomial R
    h1 : Eq (HMul.hMul u v) 1
    h2 : Eq (HMul.hMul v u) 1
    ha : Eq (HMul.hMul f ↑{ val := u, inv := v, val_inv := h1, inv_val := h2 }) g
    a b : Polynomial R
    h : Eq (HMul.hMul (HAdd.hAdd (HMul.hMul a f) (HMul.hMul b (Polynomial.derivati …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a v) (HMul.hMul b (Polynomial …
  -/
  have h3 := congr(derivative $(h1))
  /-
    case intro.mk.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    f g u v : Polynomial R
    h1 : Eq (HMul.hMul u v) 1
    h2 : Eq (HMul.hMul v u) 1
    ha : Eq (HMul.hMul f ↑{ val := u, inv := v, val_inv := h1, inv_val := h2 }) g
    a b : Polynomial R
    h : Eq (HMul.hMul (HAdd.hAdd (HMul.hMul a f) (HMul.hMul b (Polynomial.derivati …
    h3 : Eq (Polynomial.derivative (HMul.hMul u v)) (Polynomial.derivative 1)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul a v) (HMul.hMul b (Polynomial …
  -/
  simp only [← ha, derivative_mul, derivative_one] at h3 ⊢
  calc
    _ = (a * f + b * derivative f) * (u * v)
      + (b * f) * (derivative u * v + u * derivative v) := by ring1
    _ = 1 := by rw [h, h3]; ring1


theorem _root_.Associated.separable_iff {f g : R[X]}
    (ha : Associated f g) : f.Separable ↔ g.Separable := ⟨ha.separable, ha.symm.separable⟩


theorem Separable.mul_unit {f g : R[X]} (hf : f.Separable) (hg : IsUnit g) : (f * g).Separable :=
  (associated_mul_unit_right f g hg).separable hf


theorem Separable.unit_mul {f g : R[X]} (hf : IsUnit f) (hg : g.Separable) : (f * g).Separable :=
  (associated_unit_mul_right g f hf).separable hg


theorem Separable.eval₂_derivative_ne_zero [Nontrivial S] (f : R →+* S) {p : R[X]}
    (h : p.Separable) {x : S} (hx : p.eval₂ f x = 0) :
    (derivative p).eval₂ f x ≠ 0 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    p : Polynomial R
    h : p.Separable
    x : S
    hx : Eq (Polynomial.eval₂ f x p) 0
    ⊢ Ne (Polynomial.eval₂ f x (Polynomial.derivative p)) 0
  -/
  intro hx'
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    p : Polynomial R
    h : p.Separable
    x : S
    hx : Eq (Polynomial.eval₂ f x p) 0
    hx' : Eq (Polynomial.eval₂ f x (Polynomial.derivative p)) 0
    ⊢ False
  -/
  obtain ⟨a, b, e⟩ := h
  /-
    case intro.intro
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    p : Polynomial R
    x : S
    hx : Eq (Polynomial.eval₂ f x p) 0
    hx' : Eq (Polynomial.eval₂ f x (Polynomial.derivative p)) 0
    a b : Polynomial R
    e : Eq (HAdd.hAdd (HMul.hMul a p) (HMul.hMul b (Polynomial.derivative p))) 1
    ⊢ False
  -/
  apply_fun Polynomial.eval₂ f x at e
  /-
    case intro.intro
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : CommSemiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    p : Polynomial R
    x : S
    hx : Eq (Polynomial.eval₂ f x p) 0
    hx' : Eq (Polynomial.eval₂ f x (Polynomial.derivative p)) 0
    a b : Polynomial R
    e : Eq (Polynomial.eval₂ f x (HAdd.hAdd (HMul.hMul a p) (HMul.hMul b (Polynomi …
    ⊢ False
  -/
  simp only [eval₂_add, eval₂_mul, hx, mul_zero, hx', add_zero, eval₂_one, zero_ne_one] at e
  /-
    🎉 no goals
  -/


theorem Separable.aeval_derivative_ne_zero [Nontrivial S] [Algebra R S] {p : R[X]}
    (h : p.Separable) {x : S} (hx : aeval x p = 0) :
    aeval x (derivative p) ≠ 0 :=
  h.eval₂_derivative_ne_zero (algebraMap R S) hx


theorem isUnit_of_self_mul_dvd_separable {p q : R[X]} (hp : p.Separable) (hq : q * q ∣ p) :
    IsUnit q := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hp : p.Separable
    hq : Dvd.dvd (HMul.hMul q q) p
    ⊢ IsUnit q
  -/
  obtain ⟨p, rfl⟩ := hq
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    q p : Polynomial R
    hp : (HMul.hMul (HMul.hMul q q) p).Separable
    ⊢ IsUnit q
  -/
  apply isCoprime_self.mp
  have : IsCoprime (q * (q * p))
      (q * (derivative q * p + derivative q * p + q * derivative p)) := by
    simp only [← mul_assoc, mul_add]
    dsimp only [Separable] at hp
    convert hp using 1
    rw [derivative_mul, derivative_mul]
    ring
  /-
    case intro
    R : Type u
    inst✝ : CommSemiring R
    q p : Polynomial R
    hp : (HMul.hMul (HMul.hMul q q) p).Separable
    this : IsCoprime (HMul.hMul q (HMul.hMul q p)) (HMul.hMul q (HAdd.hAdd (HAdd.h …
    ⊢ IsCoprime q q
  -/
  exact IsCoprime.of_mul_right_left (IsCoprime.of_mul_left_left this)
  /-
    🎉 no goals
  -/


theorem emultiplicity_le_one_of_separable {p q : R[X]} (hq : ¬IsUnit q) (hsep : Separable p) :
    emultiplicity q p ≤ 1 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hq : Not (IsUnit q)
    hsep : p.Separable
    ⊢ LE.le (emultiplicity q p) 1
  -/
  contrapose! hq
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hsep : p.Separable
    hq : LT.lt 1 (emultiplicity q p)
    ⊢ IsUnit q
  -/
  apply isUnit_of_self_mul_dvd_separable hsep
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hsep : p.Separable
    hq : LT.lt 1 (emultiplicity q p)
    ⊢ Dvd.dvd (HMul.hMul q q) p
  -/
  rw [← sq]
  /-
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hsep : p.Separable
    hq : LT.lt 1 (emultiplicity q p)
    ⊢ Dvd.dvd (HPow.hPow q 2) p
  -/
  apply pow_dvd_of_le_emultiplicity
  /-
    case hk
    R : Type u
    inst✝ : CommSemiring R
    p q : Polynomial R
    hsep : p.Separable
    hq : LT.lt 1 (emultiplicity q p)
    ⊢ LE.le (↑2) (emultiplicity q p)
  -/
  exact Order.add_one_le_of_lt hq
  /-
    🎉 no goals
  -/


/-- A separable polynomial is square-free.

See `PerfectField.separable_iff_squarefree` for the converse when the coefficients are a perfect
field. -/
theorem Separable.squarefree {p : R[X]} (hsep : Separable p) : Squarefree p := by
  classical
  rw [squarefree_iff_emultiplicity_le_one p]
  exact fun f => or_iff_not_imp_right.mpr fun hunit => emultiplicity_le_one_of_separable hunit hsep


theorem separable_X_sub_C {x : R} : Separable (X - C x) := by
  /-
    R : Type u
    inst✝ : CommRing R
    x : R
    ⊢ (HSub.hSub Polynomial.X (Polynomial.C x)).Separable
  -/
  simpa only [sub_eq_add_neg, C_neg] using separable_X_add_C (-x)
  /-
    🎉 no goals
  -/


theorem Separable.mul {f g : R[X]} (hf : f.Separable) (hg : g.Separable) (h : IsCoprime f g) :
    (f * g).Separable := by
  /-
    R : Type u
    inst✝ : CommRing R
    f g : Polynomial R
    hf : f.Separable
    hg : g.Separable
    h : IsCoprime f g
    ⊢ (HMul.hMul f g).Separable
  -/
  rw [separable_def, derivative_mul]
  exact
    ((hf.mul_right h).add_mul_left_right _).mul_left ((h.symm.mul_right hg).mul_add_right_right _)


theorem separable_prod' {ι : Sort _} {f : ι → R[X]} {s : Finset ι} :
    (∀ x ∈ s, ∀ y ∈ s, x ≠ y → IsCoprime (f x) (f y)) →
      (∀ x ∈ s, (f x).Separable) → (∏ x ∈ s, f x).Separable := by
  classical
  exact Finset.induction_on s (fun _ _ => separable_one) fun a s has ih h1 h2 => by
    simp_rw [Finset.forall_mem_insert, forall_and] at h1 h2; rw [prod_insert has]
    exact
      h2.1.mul (ih h1.2.2 h2.2)
        (IsCoprime.prod_right fun i his => h1.1.2 i his <| Ne.symm <| ne_of_mem_of_not_mem his has)


theorem separable_prod {ι : Sort _} [Fintype ι] {f : ι → R[X]} (h1 : Pairwise (IsCoprime on f))
    (h2 : ∀ x, (f x).Separable) : (∏ x, f x).Separable :=
  separable_prod' (fun _x _hx _y _hy hxy => h1 hxy) fun x _hx => h2 x


theorem Separable.inj_of_prod_X_sub_C [Nontrivial R] {ι : Sort _} {f : ι → R} {s : Finset ι}
    (hfs : (∏ i ∈ s, (X - C (f i))).Separable) {x y : ι} (hx : x ∈ s) (hy : y ∈ s)
    (hfxy : f x = f y) : x = y := by
  classical
  by_contra hxy
  rw [← insert_erase hx, prod_insert (not_mem_erase _ _), ←
    insert_erase (mem_erase_of_ne_of_mem (Ne.symm hxy) hy), prod_insert (not_mem_erase _ _), ←
    mul_assoc, hfxy, ← sq] at hfs
  cases (hfs.of_mul_left.of_pow (not_isUnit_X_sub_C _) two_ne_zero).2


theorem Separable.injective_of_prod_X_sub_C [Nontrivial R] {ι : Sort _} [Fintype ι] {f : ι → R}
    (hfs : (∏ i, (X - C (f i))).Separable) : Function.Injective f := fun _x _y hfxy =>
  hfs.inj_of_prod_X_sub_C (mem_univ _) (mem_univ _) hfxy


theorem nodup_of_separable_prod [Nontrivial R] {s : Multiset R}
    (hs : Separable (Multiset.map (fun a => X - C a) s).prod) : s.Nodup := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    s : Multiset R
    hs : (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) s).prod. …
    ⊢ s.Nodup
  -/
  rw [Multiset.nodup_iff_ne_cons_cons]
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    s : Multiset R
    hs : (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) s).prod. …
    ⊢ ∀ (a : R) (t : Multiset R), Ne s (Multiset.cons a (Multiset.cons a t))
  -/
  rintro a t rfl
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    a : R
    t : Multiset R
    hs : (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (Multise …
    ⊢ False
  -/
  refine not_isUnit_X_sub_C a (isUnit_of_self_mul_dvd_separable hs ?_)
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    a : R
    t : Multiset R
    hs : (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) (Multise …
    ⊢ Dvd.dvd (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C a)) (HSub.hSub Poly …
  -/
  simpa only [Multiset.map_cons, Multiset.prod_cons] using mul_dvd_mul_left _ (dvd_mul_right _ _)
  /-
    🎉 no goals
  -/


/-- If `IsUnit n` in a `CommRing R`, then `X ^ n - u` is separable for any unit `u`. -/
theorem separable_X_pow_sub_C_unit {n : ℕ} (u : Rˣ) (hn : IsUnit (n : R)) :
    Separable (X ^ n - C (u : R)) := by
  /-
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C ↑u)).Separable
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    a✝ : Nontrivial R
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C ↑u)).Separable
  -/
  rcases n.eq_zero_or_pos with (rfl | hpos)
    /-
      case inl
      R : Type u
      inst✝ : CommRing R
      u : Units R
      a✝ : Nontrivial R
      hn : IsUnit ↑0
      ⊢ (HSub.hSub (HPow.hPow Polynomial.X 0) (Polynomial.C ↑u)).Separable
    -/
  · simp at hn
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    a✝ : Nontrivial R
    hpos : GT.gt n 0
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C ↑u)).Separable
  -/
  apply (separable_def' (X ^ n - C (u : R))).2
  /-
    case inr
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    a✝ : Nontrivial R
    hpos : GT.gt n 0
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a (HSub.hSub (HPow. …
  -/
  obtain ⟨n', hn'⟩ := hn.exists_left_inv
  /-
    case inr.intro
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    a✝ : Nontrivial R
    hpos : GT.gt n 0
    n' : R
    hn' : Eq (HMul.hMul n' ↑n) 1
    ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a (HSub.hSub (HPow. …
  -/
  refine ⟨-C ↑u⁻¹, C (↑u⁻¹ : R) * C n' * X, ?_⟩
  /-
    case inr.intro
    R : Type u
    inst✝ : CommRing R
    n : Nat
    u : Units R
    hn : IsUnit ↑n
    a✝ : Nontrivial R
    hpos : GT.gt n 0
    n' : R
    hn' : Eq (HMul.hMul n' ↑n) 1
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (Polynomial.C ↑(Inv.inv u))) (HSub.hSub (H …
  -/
  rw [derivative_sub, derivative_C, sub_zero, derivative_pow X n, derivative_X, mul_one]
  calc
    -C ↑u⁻¹ * (X ^ n - C ↑u) + C ↑u⁻¹ * C n' * X * (↑n * X ^ (n - 1)) =
        C (↑u⁻¹ * ↑u) - C ↑u⁻¹ * X ^ n + C ↑u⁻¹ * C (n' * ↑n) * (X * X ^ (n - 1)) := by
      simp only [C.map_mul, C_eq_natCast]
      ring
    _ = 1 := by
      simp only [Units.inv_mul, hn', C.map_one, mul_one, ← pow_succ',
        Nat.sub_add_cancel (show 1 ≤ n from hpos), sub_add_cancel]


/-- If `n = 0` in `R` and `b` is a unit, then `a * X ^ n + b * X + c` is separable. -/
theorem separable_C_mul_X_pow_add_C_mul_X_add_C
    {n : ℕ} (a b c : R) (hn : (n : R) = 0) (hb : IsUnit b) :
    (C a * X ^ n + C b * X + C c).Separable := by
  /-
    R : Type u
    inst✝ : CommRing R
    n : Nat
    a b c : R
    hn : Eq (↑n) 0
    hb : IsUnit b
    ⊢ (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n) …
  -/
  set f := C a * X ^ n + C b * X + C c
  have hderiv : derivative f = C b := by
    simp_rw [f, map_add derivative, derivative_C]
    simp [hn]
  /-
    R : Type u
    inst✝ : CommRing R
    n : Nat
    a b c : R
    hn : Eq (↑n) 0
    hb : IsUnit b
    f : Polynomial R := HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPo …
    hderiv : Eq (Polynomial.derivative f) (Polynomial.C b)
    ⊢ f.Separable
  -/
  obtain ⟨e, hb⟩ := hb.exists_left_inv
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    n : Nat
    a b c : R
    hn : Eq (↑n) 0
    hb✝ : IsUnit b
    f : Polynomial R := HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C a) (HPow.hPo …
    hderiv : Eq (Polynomial.derivative f) (Polynomial.C b)
    e : R
    hb : Eq (HMul.hMul e b) 1
    ⊢ f.Separable
  -/
  refine ⟨-derivative f, f + C e, ?_⟩
  rw [hderiv, right_distrib, ← add_assoc, neg_mul, mul_comm, neg_add_cancel, zero_add,
    ← map_mul, hb, map_one]


/-- If `R` is of characteristic `p`, `p ∣ n` and `b` is a unit,
then `a * X ^ n + b * X + c` is separable. -/
theorem separable_C_mul_X_pow_add_C_mul_X_add_C'
    (p n : ℕ) (a b c : R) [CharP R p] (hn : p ∣ n) (hb : IsUnit b) :
    (C a * X ^ n + C b * X + C c).Separable :=
  separable_C_mul_X_pow_add_C_mul_X_add_C a b c ((CharP.cast_eq_zero_iff R p n).2 hn) hb


theorem rootMultiplicity_le_one_of_separable [Nontrivial R] {p : R[X]} (hsep : Separable p)
    (x : R) : rootMultiplicity x p ≤ 1 := by
  classical
  by_cases hp : p = 0
  · simp [hp]
  rw [rootMultiplicity_eq_multiplicity, if_neg hp, ← Nat.cast_le (α := ℕ∞),
    Nat.cast_one, ← (finiteMultiplicity_X_sub_C x hp).emultiplicity_eq_multiplicity]
  apply emultiplicity_le_one_of_separable (not_isUnit_X_sub_C _) hsep


theorem count_roots_le_one [DecidableEq R] {p : R[X]} (hsep : Separable p) (x : R) :
    p.roots.count x ≤ 1 := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    p : Polynomial R
    hsep : p.Separable
    x : R
    ⊢ LE.le (Multiset.count x p.roots) 1
  -/
  rw [count_roots p]
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : DecidableEq R
    p : Polynomial R
    hsep : p.Separable
    x : R
    ⊢ LE.le (Polynomial.rootMultiplicity x p) 1
  -/
  exact rootMultiplicity_le_one_of_separable hsep x
  /-
    🎉 no goals
  -/


theorem nodup_roots {p : R[X]} (hsep : Separable p) : p.roots.Nodup := by
  classical
  exact Multiset.nodup_iff_count_le_one.mpr (count_roots_le_one hsep)


theorem separable_iff_derivative_ne_zero {f : F[X]} (hf : Irreducible f) :
    f.Separable ↔ derivative f ≠ 0 :=
  ⟨fun h1 h2 => hf.not_unit <| isCoprime_zero_right.1 <| h2 ▸ h1, fun h =>
    EuclideanDomain.isCoprime_of_dvd (mt And.right h) fun g hg1 _hg2 ⟨p, hg3⟩ hg4 =>
      let ⟨u, hu⟩ := (hf.isUnit_or_isUnit hg3).resolve_left hg1
      have : f ∣ derivative f := by
        /-
          F : Type u
          inst✝ : Field F
          f : Polynomial F
          hf : Irreducible f
          h : Ne (Polynomial.derivative f) 0
          g : Polynomial F
          hg1 : Membership.mem (nonunits (Polynomial F)) g
          _hg2 : Ne g 0
          x✝ : Dvd.dvd g f
          hg4 : Dvd.dvd g (Polynomial.derivative f)
          p : Polynomial F
          hg3 : Eq f (HMul.hMul g p)
          u : Units (Polynomial F)
          hu : Eq (↑u) p
          ⊢ Dvd.dvd f (Polynomial.derivative f)
        -/
        conv_lhs => rw [hg3, ← hu]
        /-
          F : Type u
          inst✝ : Field F
          f : Polynomial F
          hf : Irreducible f
          h : Ne (Polynomial.derivative f) 0
          g : Polynomial F
          hg1 : Membership.mem (nonunits (Polynomial F)) g
          _hg2 : Ne g 0
          x✝ : Dvd.dvd g f
          hg4 : Dvd.dvd g (Polynomial.derivative f)
          p : Polynomial F
          hg3 : Eq f (HMul.hMul g p)
          u : Units (Polynomial F)
          hu : Eq (↑u) p
          ⊢ Dvd.dvd (HMul.hMul g ↑u) (Polynomial.derivative f)
        -/
        rwa [Units.mul_right_dvd]
        /-
          🎉 no goals
        -/
      not_lt_of_le (natDegree_le_of_dvd this h) <|
        natDegree_derivative_lt <| mt derivative_of_natDegree_zero h⟩


attribute [local instance] Ideal.Quotient.field in
theorem separable_map {S} [CommRing S] [Nontrivial S] (f : F →+* S) {p : F[X]} :
    (p.map f).Separable ↔ p.Separable := by
  /-
    F : Type u
    inst✝² : Field F
    S : Type u_1
    inst✝¹ : CommRing S
    inst✝ : Nontrivial S
    f : RingHom F S
    p : Polynomial F
    ⊢ Iff (Polynomial.map f p).Separable p.Separable
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ H.map⟩
  /-
    F : Type u
    inst✝² : Field F
    S : Type u_1
    inst✝¹ : CommRing S
    inst✝ : Nontrivial S
    f : RingHom F S
    p : Polynomial F
    H : (Polynomial.map f p).Separable
    ⊢ p.Separable
  -/
  obtain ⟨m, hm⟩ := Ideal.exists_maximal S
  /-
    case intro
    F : Type u
    inst✝² : Field F
    S : Type u_1
    inst✝¹ : CommRing S
    inst✝ : Nontrivial S
    f : RingHom F S
    p : Polynomial F
    H : (Polynomial.map f p).Separable
    m : Ideal S
    hm : m.IsMaximal
    ⊢ p.Separable
  -/
  have := Separable.map H (f := Ideal.Quotient.mk m)
  /-
    case intro
    F : Type u
    inst✝² : Field F
    S : Type u_1
    inst✝¹ : CommRing S
    inst✝ : Nontrivial S
    f : RingHom F S
    p : Polynomial F
    H : (Polynomial.map f p).Separable
    m : Ideal S
    hm : m.IsMaximal
    this : (Polynomial.map (Ideal.Quotient.mk m) (Polynomial.map f p)).Separable
    ⊢ p.Separable
  -/
  rwa [map_map, separable_def, derivative_map, isCoprime_map] at this
  /-
    🎉 no goals
  -/


theorem separable_prod_X_sub_C_iff' {ι : Sort _} {f : ι → F} {s : Finset ι} :
    (∏ i ∈ s, (X - C (f i))).Separable ↔ ∀ x ∈ s, ∀ y ∈ s, f x = f y → x = y :=
  ⟨fun hfs _ hx _ hy hfxy => hfs.inj_of_prod_X_sub_C hx hy hfxy, fun H => by
    /-
      F : Type u
      inst✝ : Field F
      ι : Type u_1
      f : ι → F
      s : Finset ι
      H : ∀ (x : ι), Membership.mem s x → ∀ (y : ι), Membership.mem s y → Eq (f x) ( …
      ⊢ (s.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (f i))).Separable
    -/
    rw [← prod_attach]
    exact
      separable_prod'
        (fun x _hx y _hy hxy =>
          @pairwise_coprime_X_sub_C _ _ { x // x ∈ s } (fun x => f x)
            (fun x y hxy => Subtype.eq <| H x.1 x.2 y.1 y.2 hxy) _ _ hxy)
        fun _ _ => separable_X_sub_C⟩


theorem separable_prod_X_sub_C_iff {ι : Sort _} [Fintype ι] {f : ι → F} :
    (∏ i, (X - C (f i))).Separable ↔ Function.Injective f :=
                                          /-
                                            F : Type u
                                            inst✝¹ : Field F
                                            ι : Type u_1
                                            inst✝ : Fintype ι
                                            f : ι → F
                                            ⊢ Iff (∀ (x : ι), Membership.mem Finset.univ x → ∀ (y : ι), Membership.mem Fin …
                                          -/
  separable_prod_X_sub_C_iff'.trans <| by simp_rw [mem_univ, true_imp_iff, Function.Injective]
                                          /-
                                            🎉 no goals
                                          -/


theorem separable_or {f : F[X]} (hf : Irreducible f) :
    f.Separable ∨ ¬f.Separable ∧ ∃ g : F[X], Irreducible g ∧ expand F p g = f := by
  classical
  exact if H : derivative f = 0 then by
    rcases p.eq_zero_or_pos with (rfl | hp)
    · haveI := CharP.charP_to_charZero F
      have := natDegree_eq_zero_of_derivative_eq_zero H
      have := (natDegree_pos_iff_degree_pos.mpr <| degree_pos_of_irreducible hf).ne'
      contradiction
    haveI := isLocalHom_expand F hp
    exact
      Or.inr
        ⟨by rw [separable_iff_derivative_ne_zero hf, Classical.not_not, H], contract p f,
          Irreducible.of_map (by rwa [← expand_contract p H hp.ne'] at hf),
          expand_contract p H hp.ne'⟩
  else Or.inl <| (separable_iff_derivative_ne_zero hf).2 H


theorem exists_separable_of_irreducible {f : F[X]} (hf : Irreducible f) (hp : p ≠ 0) :
    ∃ (n : ℕ) (g : F[X]), g.Separable ∧ expand F (p ^ n) g = f := by
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : Ne p 0
    ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
  -/
  replace hp : p.Prime := (CharP.char_is_prime_or_zero F p).resolve_right hp
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : Nat.Prime p
    ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
  -/
  induction' hn : f.natDegree using Nat.strong_induction_on with N ih generalizing f
  /-
    case h
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    hp : Nat.Prime p
    N : Nat
    ih : ∀ (m : Nat), LT.lt m N → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
    f : Polynomial F
    hf : Irreducible f
    hn : Eq f.natDegree N
    ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
  -/
  rcases separable_or p hf with (h | ⟨h1, g, hg, hgf⟩)
    /-
      case h.inl
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      N : Nat
      ih : ∀ (m : Nat), LT.lt m N → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
      f : Polynomial F
      hf : Irreducible f
      hn : Eq f.natDegree N
      h : f.Separable
      ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
    -/
  · refine ⟨0, f, h, ?_⟩
    /-
      case h.inl
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      N : Nat
      ih : ∀ (m : Nat), LT.lt m N → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
      f : Polynomial F
      hf : Irreducible f
      hn : Eq f.natDegree N
      h : f.Separable
      ⊢ Eq ((Polynomial.expand F (HPow.hPow p 0)) f) f
    -/
    rw [pow_zero, expand_one]
    /-
      🎉 no goals
    -/
    /-
      case h.inr.intro.intro.intro
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      N : Nat
      ih : ∀ (m : Nat), LT.lt m N → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
      f : Polynomial F
      hf : Irreducible f
      hn : Eq f.natDegree N
      h1 : Not f.Separable
      g : Polynomial F
      hg : Irreducible g
      hgf : Eq ((Polynomial.expand F p) g) f
      ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
    -/
  · cases' N with N
      /-
        case h.inr.intro.intro.intro.zero
        F : Type u
        inst✝ : Field F
        p : Nat
        HF : CharP F p
        hp : Nat.Prime p
        f : Polynomial F
        hf : Irreducible f
        h1 : Not f.Separable
        g : Polynomial F
        hg : Irreducible g
        hgf : Eq ((Polynomial.expand F p) g) f
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
        hn : Eq f.natDegree 0
        ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
      -/
    · rw [natDegree_eq_zero_iff_degree_le_zero, degree_le_zero_iff] at hn
      /-
        case h.inr.intro.intro.intro.zero
        F : Type u
        inst✝ : Field F
        p : Nat
        HF : CharP F p
        hp : Nat.Prime p
        f : Polynomial F
        hf : Irreducible f
        h1 : Not f.Separable
        g : Polynomial F
        hg : Irreducible g
        hgf : Eq ((Polynomial.expand F p) g) f
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
        hn : Eq f (Polynomial.C (f.coeff 0))
        ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
      -/
      rw [hn, separable_C, isUnit_iff_ne_zero, Classical.not_not] at h1
      /-
        case h.inr.intro.intro.intro.zero
        F : Type u
        inst✝ : Field F
        p : Nat
        HF : CharP F p
        hp : Nat.Prime p
        f : Polynomial F
        hf : Irreducible f
        h1 : Eq (f.coeff 0) 0
        g : Polynomial F
        hg : Irreducible g
        hgf : Eq ((Polynomial.expand F p) g) f
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
        hn : Eq f (Polynomial.C (f.coeff 0))
        ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
      -/
      have hf0 : f ≠ 0 := hf.ne_zero
      /-
        case h.inr.intro.intro.intro.zero
        F : Type u
        inst✝ : Field F
        p : Nat
        HF : CharP F p
        hp : Nat.Prime p
        f : Polynomial F
        hf : Irreducible f
        h1 : Eq (f.coeff 0) 0
        g : Polynomial F
        hg : Irreducible g
        hgf : Eq ((Polynomial.expand F p) g) f
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
        hn : Eq f (Polynomial.C (f.coeff 0))
        hf0 : Ne f 0
        ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
      -/
      rw [h1, C_0] at hn
      /-
        case h.inr.intro.intro.intro.zero
        F : Type u
        inst✝ : Field F
        p : Nat
        HF : CharP F p
        hp : Nat.Prime p
        f : Polynomial F
        hf : Irreducible f
        h1 : Eq (f.coeff 0) 0
        g : Polynomial F
        hg : Irreducible g
        hgf : Eq ((Polynomial.expand F p) g) f
        ih : ∀ (m : Nat), LT.lt m 0 → ∀ {f : Polynomial F}, Irreducible f → Eq f.natDe …
        hn : Eq f 0
        hf0 : Ne f 0
        ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
      -/
      exact absurd hn hf0
      /-
        🎉 no goals
      -/
    /-
      case h.inr.intro.intro.intro.succ
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      f : Polynomial F
      hf : Irreducible f
      h1 : Not f.Separable
      g : Polynomial F
      hg : Irreducible g
      hgf : Eq ((Polynomial.expand F p) g) f
      N : Nat
      ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd N 1) → ∀ {f : Polynomial F}, Irreducible  …
      hn : Eq f.natDegree (HAdd.hAdd N 1)
      ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
    -/
    have hg1 : g.natDegree * p = N.succ := by rwa [← natDegree_expand, hgf]
    have hg2 : g.natDegree ≠ 0 := by
      intro this
      rw [this, zero_mul] at hg1
      cases hg1
    have hg3 : g.natDegree < N.succ := by
      rw [← mul_one g.natDegree, ← hg1]
      exact Nat.mul_lt_mul_of_pos_left hp.one_lt hg2.bot_lt
    /-
      case h.inr.intro.intro.intro.succ
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      f : Polynomial F
      hf : Irreducible f
      h1 : Not f.Separable
      g : Polynomial F
      hg : Irreducible g
      hgf : Eq ((Polynomial.expand F p) g) f
      N : Nat
      ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd N 1) → ∀ {f : Polynomial F}, Irreducible  …
      hn : Eq f.natDegree (HAdd.hAdd N 1)
      hg1 : Eq (HMul.hMul g.natDegree p) N.succ
      hg2 : Ne g.natDegree 0
      hg3 : LT.lt g.natDegree N.succ
      ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
    -/
    rcases ih _ hg3 hg rfl with ⟨n, g, hg4, rfl⟩
    /-
      case h.inr.intro.intro.intro.succ.intro.intro.intro
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      f : Polynomial F
      hf : Irreducible f
      h1 : Not f.Separable
      N : Nat
      ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd N 1) → ∀ {f : Polynomial F}, Irreducible  …
      hn : Eq f.natDegree (HAdd.hAdd N 1)
      n : Nat
      g : Polynomial F
      hg4 : g.Separable
      hg : Irreducible ((Polynomial.expand F (HPow.hPow p n)) g)
      hgf : Eq ((Polynomial.expand F p) ((Polynomial.expand F (HPow.hPow p n)) g)) f
      hg1 : Eq (HMul.hMul ((Polynomial.expand F (HPow.hPow p n)) g).natDegree p) N.s …
      hg2 : Ne ((Polynomial.expand F (HPow.hPow p n)) g).natDegree 0
      hg3 : LT.lt ((Polynomial.expand F (HPow.hPow p n)) g).natDegree N.succ
      ⊢ Exists fun n => Exists fun g => And g.Separable (Eq ((Polynomial.expand F (H …
    -/
    refine ⟨n + 1, g, hg4, ?_⟩
    /-
      case h.inr.intro.intro.intro.succ.intro.intro.intro
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : Nat.Prime p
      f : Polynomial F
      hf : Irreducible f
      h1 : Not f.Separable
      N : Nat
      ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd N 1) → ∀ {f : Polynomial F}, Irreducible  …
      hn : Eq f.natDegree (HAdd.hAdd N 1)
      n : Nat
      g : Polynomial F
      hg4 : g.Separable
      hg : Irreducible ((Polynomial.expand F (HPow.hPow p n)) g)
      hgf : Eq ((Polynomial.expand F p) ((Polynomial.expand F (HPow.hPow p n)) g)) f
      hg1 : Eq (HMul.hMul ((Polynomial.expand F (HPow.hPow p n)) g).natDegree p) N.s …
      hg2 : Ne ((Polynomial.expand F (HPow.hPow p n)) g).natDegree 0
      hg3 : LT.lt ((Polynomial.expand F (HPow.hPow p n)) g).natDegree N.succ
      ⊢ Eq ((Polynomial.expand F (HPow.hPow p (HAdd.hAdd n 1))) g) f
    -/
    rw [← hgf, expand_expand, pow_succ']
    /-
      🎉 no goals
    -/


theorem isUnit_or_eq_zero_of_separable_expand {f : F[X]} (n : ℕ) (hp : 0 < p)
    (hf : (expand F (p ^ n) f).Separable) : IsUnit f ∨ n = 0 := by
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hf : ((Polynomial.expand F (HPow.hPow p n)) f).Separable
    ⊢ Or (IsUnit f) (Eq n 0)
  -/
  rw [or_iff_not_imp_right]
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hf : ((Polynomial.expand F (HPow.hPow p n)) f).Separable
    ⊢ Not (Eq n 0) → IsUnit f
  -/
  rintro hn : n ≠ 0
  have hf2 : derivative (expand F (p ^ n) f) = 0 := by
    rw [derivative_expand, Nat.cast_pow, CharP.cast_eq_zero, zero_pow hn, zero_mul, mul_zero]
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hf : ((Polynomial.expand F (HPow.hPow p n)) f).Separable
    hn : Ne n 0
    hf2 : Eq (Polynomial.derivative ((Polynomial.expand F (HPow.hPow p n)) f)) 0
    ⊢ IsUnit f
  -/
  rw [separable_def, hf2, isCoprime_zero_right, isUnit_iff] at hf
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hf : Exists fun r => And (IsUnit r) (Eq (Polynomial.C r) ((Polynomial.expand F …
    hn : Ne n 0
    hf2 : Eq (Polynomial.derivative ((Polynomial.expand F (HPow.hPow p n)) f)) 0
    ⊢ IsUnit f
  -/
  rcases hf with ⟨r, hr, hrf⟩
  /-
    case intro.intro
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hn : Ne n 0
    hf2 : Eq (Polynomial.derivative ((Polynomial.expand F (HPow.hPow p n)) f)) 0
    r : F
    hr : IsUnit r
    hrf : Eq (Polynomial.C r) ((Polynomial.expand F (HPow.hPow p n)) f)
    ⊢ IsUnit f
  -/
  rw [eq_comm, expand_eq_C (pow_pos hp _)] at hrf
  /-
    case intro.intro
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    n : Nat
    hp : LT.lt 0 p
    hn : Ne n 0
    hf2 : Eq (Polynomial.derivative ((Polynomial.expand F (HPow.hPow p n)) f)) 0
    r : F
    hr : IsUnit r
    hrf : Eq f (Polynomial.C r)
    ⊢ IsUnit f
  -/
  rwa [hrf, isUnit_C]
  /-
    🎉 no goals
  -/


theorem unique_separable_of_irreducible {f : F[X]} (hf : Irreducible f) (hp : 0 < p) (n₁ : ℕ)
    (g₁ : F[X]) (hg₁ : g₁.Separable) (hgf₁ : expand F (p ^ n₁) g₁ = f) (n₂ : ℕ) (g₂ : F[X])
    (hg₂ : g₂.Separable) (hgf₂ : expand F (p ^ n₂) g₂ = f) : n₁ = n₂ ∧ g₁ = g₂ := by
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ : Nat
    g₁ : Polynomial F
    hg₁ : g₁.Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
    n₂ : Nat
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    hgf₂ : Eq ((Polynomial.expand F (HPow.hPow p n₂)) g₂) f
    ⊢ And (Eq n₁ n₂) (Eq g₁ g₂)
  -/
  revert g₁ g₂
  /-
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ n₂ : Nat
    ⊢ ∀ (g₁ : Polynomial F), g₁.Separable → Eq ((Polynomial.expand F (HPow.hPow p  …
  -/
  wlog hn : n₁ ≤ n₂
    /-
      case inr
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      f : Polynomial F
      hf : Irreducible f
      hp : LT.lt 0 p
      n₁ n₂ : Nat
      this : ∀ {F : Type u} [inst : Field F] (p : Nat) [HF : CharP F p] {f : Polynom …
      hn : Not (LE.le n₁ n₂)
      ⊢ ∀ (g₁ : Polynomial F), g₁.Separable → Eq ((Polynomial.expand F (HPow.hPow p  …
    -/
  · intro g₁ hg₁ Hg₁ g₂ hg₂ Hg₂
    /-
      case inr
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      f : Polynomial F
      hf : Irreducible f
      hp : LT.lt 0 p
      n₁ n₂ : Nat
      this : ∀ {F : Type u} [inst : Field F] (p : Nat) [HF : CharP F p] {f : Polynom …
      hn : Not (LE.le n₁ n₂)
      g₁ : Polynomial F
      hg₁ : g₁.Separable
      Hg₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
      g₂ : Polynomial F
      hg₂ : g₂.Separable
      Hg₂ : Eq ((Polynomial.expand F (HPow.hPow p n₂)) g₂) f
      ⊢ And (Eq n₁ n₂) (Eq g₁ g₂)
    -/
    simpa only [eq_comm] using this p hf hp n₂ n₁ (le_of_not_le hn) g₂ hg₂ Hg₂ g₁ hg₁ Hg₁
    /-
      🎉 no goals
    -/
  /-
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ n₂ : Nat
    hn : LE.le n₁ n₂
    ⊢ ∀ (g₁ : Polynomial F), g₁.Separable → Eq ((Polynomial.expand F (HPow.hPow p  …
  -/
  have hf0 : f ≠ 0 := hf.ne_zero
  /-
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ n₂ : Nat
    hn : LE.le n₁ n₂
    hf0 : Ne f 0
    ⊢ ∀ (g₁ : Polynomial F), g₁.Separable → Eq ((Polynomial.expand F (HPow.hPow p  …
  -/
  intros g₁ hg₁ hgf₁ g₂ hg₂ hgf₂
  /-
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ n₂ : Nat
    hn : LE.le n₁ n₂
    hf0 : Ne f 0
    g₁ : Polynomial F
    hg₁ : g₁.Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    hgf₂ : Eq ((Polynomial.expand F (HPow.hPow p n₂)) g₂) f
    ⊢ And (Eq n₁ n₂) (Eq g₁ g₂)
  -/
  rw [le_iff_exists_add] at hn
  /-
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ n₂ : Nat
    hn : Exists fun c => Eq n₂ (HAdd.hAdd n₁ c)
    hf0 : Ne f 0
    g₁ : Polynomial F
    hg₁ : g₁.Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    hgf₂ : Eq ((Polynomial.expand F (HPow.hPow p n₂)) g₂) f
    ⊢ And (Eq n₁ n₂) (Eq g₁ g₂)
  -/
  rcases hn with ⟨k, rfl⟩
  /-
    case intro
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ : Nat
    hf0 : Ne f 0
    g₁ : Polynomial F
    hg₁ : g₁.Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    k : Nat
    hgf₂ : Eq ((Polynomial.expand F (HPow.hPow p (HAdd.hAdd n₁ k))) g₂) f
    ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq g₁ g₂)
  -/
  rw [← hgf₁, pow_add, expand_mul, expand_inj (pow_pos hp n₁)] at hgf₂
  /-
    case intro
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ : Nat
    hf0 : Ne f 0
    g₁ : Polynomial F
    hg₁ : g₁.Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) g₁) f
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    k : Nat
    hgf₂ : Eq ((Polynomial.expand F (HPow.hPow p k)) g₂) g₁
    ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq g₁ g₂)
  -/
  subst hgf₂
  /-
    case intro
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    f : Polynomial F
    hf : Irreducible f
    hp : LT.lt 0 p
    n₁ : Nat
    hf0 : Ne f 0
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    k : Nat
    hg₁ : ((Polynomial.expand F (HPow.hPow p k)) g₂).Separable
    hgf₁ : Eq ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow. …
    ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) g₂)  …
  -/
  subst hgf₁
  /-
    case intro
    F✝ : Type u
    inst✝¹ : Field F✝
    p✝ : Nat
    F : Type u
    inst✝ : Field F
    p : Nat
    HF : CharP F p
    hp : LT.lt 0 p
    n₁ : Nat
    g₂ : Polynomial F
    hg₂ : g₂.Separable
    k : Nat
    hg₁ : ((Polynomial.expand F (HPow.hPow p k)) g₂).Separable
    hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
    hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
    ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) g₂)  …
  -/
  rcases isUnit_or_eq_zero_of_separable_expand p k hp hg₁ with (h | rfl)
    /-
      case intro.inl
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ : Nat
      g₂ : Polynomial F
      hg₂ : g₂.Separable
      k : Nat
      hg₁ : ((Polynomial.expand F (HPow.hPow p k)) g₂).Separable
      hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      h : IsUnit g₂
      ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) g₂)  …
    -/
  · rw [isUnit_iff] at h
    /-
      case intro.inl
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ : Nat
      g₂ : Polynomial F
      hg₂ : g₂.Separable
      k : Nat
      hg₁ : ((Polynomial.expand F (HPow.hPow p k)) g₂).Separable
      hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      h : Exists fun r => And (IsUnit r) (Eq (Polynomial.C r) g₂)
      ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) g₂)  …
    -/
    rcases h with ⟨r, hr, rfl⟩
    /-
      case intro.inl.intro.intro
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ k : Nat
      r : F
      hr : IsUnit r
      hg₂ : (Polynomial.C r).Separable
      hg₁ : ((Polynomial.expand F (HPow.hPow p k)) (Polynomial.C r)).Separable
      hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) (Pol …
    -/
    simp_rw [expand_C] at hf
    /-
      case intro.inl.intro.intro
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ k : Nat
      r : F
      hr : IsUnit r
      hg₂ : (Polynomial.C r).Separable
      hg₁ : ((Polynomial.expand F (HPow.hPow p k)) (Polynomial.C r)).Separable
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      hf : Irreducible (Polynomial.C r)
      ⊢ And (Eq n₁ (HAdd.hAdd n₁ k)) (Eq ((Polynomial.expand F (HPow.hPow p k)) (Pol …
    -/
    exact absurd (isUnit_C.2 hr) hf.1
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ : Nat
      g₂ : Polynomial F
      hg₂ : g₂.Separable
      hg₁ : ((Polynomial.expand F (HPow.hPow p 0)) g₂).Separable
      hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      ⊢ And (Eq n₁ (HAdd.hAdd n₁ 0)) (Eq ((Polynomial.expand F (HPow.hPow p 0)) g₂)  …
    -/
  · rw [add_zero, pow_zero, expand_one]
    /-
      case intro.inr
      F✝ : Type u
      inst✝¹ : Field F✝
      p✝ : Nat
      F : Type u
      inst✝ : Field F
      p : Nat
      HF : CharP F p
      hp : LT.lt 0 p
      n₁ : Nat
      g₂ : Polynomial F
      hg₂ : g₂.Separable
      hg₁ : ((Polynomial.expand F (HPow.hPow p 0)) g₂).Separable
      hf : Irreducible ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F …
      hf0 : Ne ((Polynomial.expand F (HPow.hPow p n₁)) ((Polynomial.expand F (HPow.h …
      ⊢ And (Eq n₁ n₁) (Eq g₂ g₂)
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> rfl
                    /-
                      🎉 no goals
                    -/


/-- If `n ≠ 0` in `F`, then `X ^ n - a` is separable for any `a ≠ 0`. -/
theorem separable_X_pow_sub_C {n : ℕ} (a : F) (hn : (n : F) ≠ 0) (ha : a ≠ 0) :
    Separable (X ^ n - C a) :=
  separable_X_pow_sub_C_unit (Units.mk0 a ha) (IsUnit.mk0 (n : F) hn)


/-- If `F` is of characteristic `p` and `p ∤ n`, then `X ^ n - a` is separable for any `a ≠ 0`. -/
theorem separable_X_pow_sub_C' (p n : ℕ) (a : F) [CharP F p] (hn : ¬p ∣ n) (ha : a ≠ 0) :
    Separable (X ^ n - C a) :=
                              /-
                                F : Type u
                                inst✝¹ : Field F
                                p n : Nat
                                a : F
                                inst✝ : CharP F p
                                hn : Not (Dvd.dvd p n)
                                ha : Ne a 0
                                ⊢ Ne (↑n) 0
                              -/
  separable_X_pow_sub_C a (by rwa [← CharP.cast_eq_zero_iff F p n] at hn) ha
                              /-
                                🎉 no goals
                              -/

-- this can possibly be strengthened to making `separable_X_pow_sub_C_unit` a
-- bi-implication, but it is nontrivial!

/-- In a field `F`, `X ^ n - 1` is separable iff `↑n ≠ 0`. -/
theorem X_pow_sub_one_separable_iff {n : ℕ} : (X ^ n - 1 : F[X]).Separable ↔ (n : F) ≠ 0 := by
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    ⊢ Iff (HSub.hSub (HPow.hPow Polynomial.X n) 1).Separable (Ne (↑n) 0)
  -/
  refine ⟨?_, fun h => separable_X_pow_sub_C_unit 1 (IsUnit.mk0 _ h)⟩
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) 1).Separable → Ne (↑n) 0
  -/
  rw [separable_def', derivative_sub, derivative_X_pow, derivative_one, sub_zero]
  -- Suppose `(n : F) = 0`, then the derivative is `0`, so `X ^ n - 1` is a unit, contradiction.
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    ⊢ (Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a (HSub.hSub (HPow …
  -/
  rintro (h : IsCoprime _ _) hn'
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    h : IsCoprime (HSub.hSub (HPow.hPow Polynomial.X n) 1) (HMul.hMul (Polynomial. …
    hn' : Eq (↑n) 0
    ⊢ False
  -/
  rw [hn', C_0, zero_mul, isCoprime_zero_right] at h
  /-
    F : Type u
    inst✝ : Field F
    n : Nat
    h : IsUnit (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    hn' : Eq (↑n) 0
    ⊢ False
  -/
  exact not_isUnit_X_pow_sub_one F n h
  /-
    🎉 no goals
  -/


theorem card_rootSet_eq_natDegree [Algebra F K] {p : F[X]} (hsep : p.Separable)
    (hsplit : Splits (algebraMap F K) p) : Fintype.card (p.rootSet K) = p.natDegree := by
  classical
  simp_rw [rootSet_def, Finset.coe_sort_coe, Fintype.card_coe]
  rw [Multiset.toFinset_card_of_nodup (nodup_roots hsep.map), ← natDegree_eq_card_roots hsplit]


/-- If a non-zero polynomial splits, then it has no repeated roots on that field
if and only if it is separable. -/
theorem nodup_roots_iff_of_splits {f : F[X]} (hf : f ≠ 0) (h : f.Splits (RingHom.id F)) :
    f.roots.Nodup ↔ f.Separable := by
  classical
  refine ⟨(fun hnsep ↦ ?_).mtr, nodup_roots⟩
  rw [Separable, ← gcd_isUnit_iff, isUnit_iff_degree_eq_zero] at hnsep
  obtain ⟨x, hx⟩ := exists_root_of_splits _
    (splits_of_splits_of_dvd _ hf h (gcd_dvd_left f _)) hnsep
  simp_rw [Multiset.nodup_iff_count_le_one, not_forall, not_le]
  exact ⟨x, ((one_lt_rootMultiplicity_iff_isRoot_gcd hf).2 hx).trans_eq f.count_roots.symm⟩


/-- If a non-zero polynomial over `F` splits in `K`, then it has no repeated roots on `K`
if and only if it is separable. -/
@[stacks 09H3 "Here we only require `f` splits instead of `K` is algebraically closed."]
theorem nodup_aroots_iff_of_splits [Algebra F K] {f : F[X]} (hf : f ≠ 0)
    (h : f.Splits (algebraMap F K)) : (f.aroots K).Nodup ↔ f.Separable := by
  /-
    F : Type u
    inst✝² : Field F
    K : Type v
    inst✝¹ : Field K
    inst✝ : Algebra F K
    f : Polynomial F
    hf : Ne f 0
    h : Polynomial.Splits (algebraMap F K) f
    ⊢ Iff (f.aroots K).Nodup f.Separable
  -/
  rw [← (algebraMap F K).id_comp, ← splits_map_iff] at h
  /-
    F : Type u
    inst✝² : Field F
    K : Type v
    inst✝¹ : Field K
    inst✝ : Algebra F K
    f : Polynomial F
    hf : Ne f 0
    h : Polynomial.Splits (RingHom.id K) (Polynomial.map (algebraMap F K) f)
    ⊢ Iff (f.aroots K).Nodup f.Separable
  -/
  rw [nodup_roots_iff_of_splits (map_ne_zero hf) h, separable_map]
  /-
    🎉 no goals
  -/


theorem card_rootSet_eq_natDegree_iff_of_splits [Algebra F K] {f : F[X]} (hf : f ≠ 0)
    (h : f.Splits (algebraMap F K)) : Fintype.card (f.rootSet K) = f.natDegree ↔ f.Separable := by
  classical
  simp_rw [rootSet_def, Finset.coe_sort_coe, Fintype.card_coe, natDegree_eq_card_roots h,
    Multiset.toFinset_card_eq_card_iff_nodup, nodup_aroots_iff_of_splits hf h]


theorem eq_X_sub_C_of_separable_of_root_eq {x : F} {h : F[X]} (h_sep : h.Separable)
    (h_root : h.eval x = 0) (h_splits : Splits i h) (h_roots : ∀ y ∈ (h.map i).roots, y = i x) :
    h = C (leadingCoeff h) * (X - C x) := by
  have h_ne_zero : h ≠ 0 := by
    rintro rfl
    exact not_separable_zero h_sep
  /-
    F : Type u
    inst✝¹ : Field F
    K : Type v
    inst✝ : Field K
    i : RingHom F K
    x : F
    h : Polynomial F
    h_sep : h.Separable
    h_root : Eq (Polynomial.eval x h) 0
    h_splits : Polynomial.Splits i h
    h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
    h_ne_zero : Ne h 0
    ⊢ Eq h (HMul.hMul (Polynomial.C h.leadingCoeff) (HSub.hSub Polynomial.X (Polyn …
  -/
  apply Polynomial.eq_X_sub_C_of_splits_of_single_root i h_splits
  /-
    F : Type u
    inst✝¹ : Field F
    K : Type v
    inst✝ : Field K
    i : RingHom F K
    x : F
    h : Polynomial F
    h_sep : h.Separable
    h_root : Eq (Polynomial.eval x h) 0
    h_splits : Polynomial.Splits i h
    h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
    h_ne_zero : Ne h 0
    ⊢ Eq (Polynomial.map i h).roots (Singleton.singleton (i x))
  -/
  apply Finset.mk.inj
    /-
      case x
      F : Type u
      inst✝¹ : Field F
      K : Type v
      inst✝ : Field K
      i : RingHom F K
      x : F
      h : Polynomial F
      h_sep : h.Separable
      h_root : Eq (Polynomial.eval x h) 0
      h_splits : Polynomial.Splits i h
      h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
      h_ne_zero : Ne h 0
      ⊢ Eq { val := (Polynomial.map i h).roots, nodup := ?nodup } { val := Singleton …
    -/
  · change _ = {i x}
    /-
      case x
      F : Type u
      inst✝¹ : Field F
      K : Type v
      inst✝ : Field K
      i : RingHom F K
      x : F
      h : Polynomial F
      h_sep : h.Separable
      h_root : Eq (Polynomial.eval x h) 0
      h_splits : Polynomial.Splits i h
      h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
      h_ne_zero : Ne h 0
      ⊢ Eq { val := (Polynomial.map i h).roots, nodup := ?nodup } (Singleton.singlet …
    -/
    rw [Finset.eq_singleton_iff_unique_mem]
    /-
      case x
      F : Type u
      inst✝¹ : Field F
      K : Type v
      inst✝ : Field K
      i : RingHom F K
      x : F
      h : Polynomial F
      h_sep : h.Separable
      h_root : Eq (Polynomial.eval x h) 0
      h_splits : Polynomial.Splits i h
      h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
      h_ne_zero : Ne h 0
      ⊢ And (Membership.mem { val := (Polynomial.map i h).roots, nodup := ?nodup } ( …
    -/
    constructor
      /-
        case x.left
        F : Type u
        inst✝¹ : Field F
        K : Type v
        inst✝ : Field K
        i : RingHom F K
        x : F
        h : Polynomial F
        h_sep : h.Separable
        h_root : Eq (Polynomial.eval x h) 0
        h_splits : Polynomial.Splits i h
        h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
        h_ne_zero : Ne h 0
        ⊢ Membership.mem { val := (Polynomial.map i h).roots, nodup := ?nodup } (i x)
      -/
    · apply Finset.mem_mk.mpr
        /-
          case x.left
          F : Type u
          inst✝¹ : Field F
          K : Type v
          inst✝ : Field K
          i : RingHom F K
          x : F
          h : Polynomial F
          h_sep : h.Separable
          h_root : Eq (Polynomial.eval x h) 0
          h_splits : Polynomial.Splits i h
          h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
          h_ne_zero : Ne h 0
          ⊢ Membership.mem (Polynomial.map i h).roots (i x)
        -/
      · rw [mem_roots (show h.map i ≠ 0 from map_ne_zero h_ne_zero)]
        /-
          case x.left
          F : Type u
          inst✝¹ : Field F
          K : Type v
          inst✝ : Field K
          i : RingHom F K
          x : F
          h : Polynomial F
          h_sep : h.Separable
          h_root : Eq (Polynomial.eval x h) 0
          h_splits : Polynomial.Splits i h
          h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
          h_ne_zero : Ne h 0
          ⊢ (Polynomial.map i h).IsRoot (i x)
        -/
        rw [IsRoot.def, ← eval₂_eq_eval_map, eval₂_hom, h_root]
        /-
          case x.left
          F : Type u
          inst✝¹ : Field F
          K : Type v
          inst✝ : Field K
          i : RingHom F K
          x : F
          h : Polynomial F
          h_sep : h.Separable
          h_root : Eq (Polynomial.eval x h) 0
          h_splits : Polynomial.Splits i h
          h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
          h_ne_zero : Ne h 0
          ⊢ Eq (i 0) 0
        -/
        exact RingHom.map_zero i
        /-
          🎉 no goals
        -/
        /-
          case nodup
          F : Type u
          inst✝¹ : Field F
          K : Type v
          inst✝ : Field K
          i : RingHom F K
          x : F
          h : Polynomial F
          h_sep : h.Separable
          h_root : Eq (Polynomial.eval x h) 0
          h_splits : Polynomial.Splits i h
          h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
          h_ne_zero : Ne h 0
          ⊢ (Polynomial.map i h).roots.Nodup
        -/
      · exact nodup_roots (Separable.map h_sep)
        /-
          🎉 no goals
        -/
      /-
        case x.right
        F : Type u
        inst✝¹ : Field F
        K : Type v
        inst✝ : Field K
        i : RingHom F K
        x : F
        h : Polynomial F
        h_sep : h.Separable
        h_root : Eq (Polynomial.eval x h) 0
        h_splits : Polynomial.Splits i h
        h_roots : ∀ (y : K), Membership.mem (Polynomial.map i h).roots y → Eq y (i x)
        h_ne_zero : Ne h 0
        ⊢ ∀ (x_1 : K), Membership.mem { val := (Polynomial.map i h).roots, nodup := ⋯  …
      -/
    · exact h_roots
      /-
        🎉 no goals
      -/


theorem exists_finset_of_splits (i : F →+* K) {f : F[X]} (sep : Separable f) (sp : Splits i f) :
    ∃ s : Finset K, f.map i = C (i f.leadingCoeff) * s.prod fun a : K => X - C a := by
  classical
  obtain ⟨s, h⟩ := (splits_iff_exists_multiset _).1 sp
  use s.toFinset
  rw [h, Finset.prod_eq_multiset_prod, ← Multiset.toFinset_eq]
  apply nodup_of_separable_prod
  apply Separable.of_mul_right
  rw [← h]
  exact sep.map


theorem _root_.Irreducible.separable [CharZero F] {f : F[X]} (hf : Irreducible f) :
    f.Separable := by
  /-
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    ⊢ f.Separable
  -/
  rw [separable_iff_derivative_ne_zero hf, Ne, ← degree_eq_bot, degree_derivative_eq]
    /-
      F : Type u
      inst✝¹ : Field F
      inst✝ : CharZero F
      f : Polynomial F
      hf : Irreducible f
      ⊢ Not (Eq (↑(HSub.hSub f.natDegree 1)) Bot.bot)
    -/
  · rintro ⟨⟩
    /-
      🎉 no goals
    -/
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    ⊢ LT.lt 0 f.natDegree
  -/
  rw [pos_iff_ne_zero, Ne, natDegree_eq_zero_iff_degree_le_zero, degree_le_zero_iff]
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    ⊢ Not (Eq f (Polynomial.C (f.coeff 0)))
  -/
  refine fun hf1 => hf.not_unit ?_
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    hf1 : Eq f (Polynomial.C (f.coeff 0))
    ⊢ IsUnit f
  -/
  rw [hf1, isUnit_C, isUnit_iff_ne_zero]
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    hf1 : Eq f (Polynomial.C (f.coeff 0))
    ⊢ Ne (f.coeff 0) 0
  -/
  intro hf2
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    hf1 : Eq f (Polynomial.C (f.coeff 0))
    hf2 : Eq (f.coeff 0) 0
    ⊢ False
  -/
  rw [hf2, C_0] at hf1
  /-
    case hp
    F : Type u
    inst✝¹ : Field F
    inst✝ : CharZero F
    f : Polynomial F
    hf : Irreducible f
    hf1 : Eq f 0
    hf2 : Eq (f.coeff 0) 0
    ⊢ False
  -/
  exact absurd hf1 hf.ne_zero
  /-
    🎉 no goals
  -/


variable {K} in
/--
An element `x` of an algebra `K` over a commutative ring `F` is said to be *separable*, if its
minimal polynomial over `K` is separable. Note that the minimal polynomial of any element not
integral over `F` is defined to be `0`, which is not a separable polynomial.
-/
@[stacks 09H1 "second part"]
def IsSeparable (x : K) : Prop := Polynomial.Separable (minpoly F x)


/-- Typeclass for separable field extension: `K` is a separable field extension of `F` iff
the minimal polynomial of every `x : K` is separable. This implies that `K/F` is an algebraic
extension, because the minimal polynomial of a non-integral element is `0`, which is not
separable.

We define this for general (commutative) rings and only assume `F` and `K` are fields if this
is needed for a proof. -/
@[mk_iff isSeparable_def, stacks 09H1 "third part"]
protected class Algebra.IsSeparable : Prop where
  isSeparable' : ∀ x : K, IsSeparable F x


theorem Algebra.IsSeparable.isSeparable [Algebra.IsSeparable F K] : ∀ x : K, IsSeparable F x :=
  Algebra.IsSeparable.isSeparable'


variable {F} in
/-- If the minimal polynomial of `x : K` over `F` is separable, then `x` is integral over `F`,
because the minimal polynomial of a non-integral element is `0`, which is not separable. -/
theorem IsSeparable.isIntegral {x : K} (h : IsSeparable F x) : IsIntegral F x := by
  /-
    F : Type u_1
    K : Type u_3
    inst✝² : CommRing F
    inst✝¹ : Ring K
    inst✝ : Algebra F K
    x : K
    h : IsSeparable F x
    ⊢ IsIntegral F x
  -/
  cases subsingleton_or_nontrivial F
    /-
      case inl
      F : Type u_1
      K : Type u_3
      inst✝² : CommRing F
      inst✝¹ : Ring K
      inst✝ : Algebra F K
      x : K
      h : IsSeparable F x
      h✝ : Subsingleton F
      ⊢ IsIntegral F x
    -/
  · haveI := Module.subsingleton F K
    /-
      case inl
      F : Type u_1
      K : Type u_3
      inst✝² : CommRing F
      inst✝¹ : Ring K
      inst✝ : Algebra F K
      x : K
      h : IsSeparable F x
      h✝ : Subsingleton F
      this : Subsingleton K
      ⊢ IsIntegral F x
    -/
    exact ⟨1, monic_one, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_1
      K : Type u_3
      inst✝² : CommRing F
      inst✝¹ : Ring K
      inst✝ : Algebra F K
      x : K
      h : IsSeparable F x
      h✝ : Nontrivial F
      ⊢ IsIntegral F x
    -/
  · exact of_not_not (h.ne_zero <| minpoly.eq_zero ·)
    /-
      🎉 no goals
    -/


theorem Algebra.IsSeparable.isIntegral [Algebra.IsSeparable F K] : ∀ x : K, IsIntegral F x :=
  fun x ↦ _root_.IsSeparable.isIntegral (Algebra.IsSeparable.isSeparable F x)


variable (K) in
instance Algebra.IsSeparable.isAlgebraic [Nontrivial F] [Algebra.IsSeparable F K] :
    Algebra.IsAlgebraic F K :=
  ⟨fun x ↦ (Algebra.IsSeparable.isIntegral F x).isAlgebraic⟩


theorem Algebra.isSeparable_iff :
    Algebra.IsSeparable F K ↔ ∀ x : K, IsIntegral F x ∧ IsSeparable F x :=
  ⟨fun _ x => ⟨Algebra.IsSeparable.isIntegral F x, Algebra.IsSeparable.isSeparable F x⟩,
    fun h => ⟨fun x => (h x).2⟩⟩


/-- Transfer `IsSeparable` across an `AlgEquiv`. -/
theorem AlgEquiv.isSeparable_iff {x : K} : IsSeparable F (e x) ↔ IsSeparable F x := by
  /-
    F : Type u_1
    K : Type u_3
    inst✝⁴ : CommRing F
    inst✝³ : Ring K
    inst✝² : Algebra F K
    E : Type u_4
    inst✝¹ : Ring E
    inst✝ : Algebra F E
    e : AlgEquiv F K E
    x : K
    ⊢ Iff (IsSeparable F (e x)) (IsSeparable F x)
  -/
  simp only [IsSeparable, minpoly.algEquiv_eq e x]
  /-
    🎉 no goals
  -/


/-- Transfer `Algebra.IsSeparable` across an `AlgEquiv`. -/
theorem AlgEquiv.Algebra.isSeparable [Algebra.IsSeparable F K] : Algebra.IsSeparable F E :=
  ⟨fun _ ↦ e.symm.isSeparable_iff.mp (Algebra.IsSeparable.isSeparable _ _)⟩


@[deprecated (since := "2024-08-06")]
alias AlgEquiv.isSeparable := AlgEquiv.Algebra.isSeparable


theorem AlgEquiv.Algebra.isSeparable_iff : Algebra.IsSeparable F K ↔ Algebra.IsSeparable F E :=
  ⟨fun _ ↦ AlgEquiv.Algebra.isSeparable e, fun _ ↦ AlgEquiv.Algebra.isSeparable e.symm⟩


/-- If `E / L / F` is a scalar tower and `x : E` is separable over `F`, then it's also separable
over `L`. -/
@[stacks 09H2 "first part"]
theorem IsSeparable.tower_top
    {x : E} (h : IsSeparable F x) : IsSeparable L x :=
  h.map.of_dvd (minpoly.dvd_map_of_isScalarTower _ _ _)


variable (F E) in
/-- If `E / K / F` is an extension tower, `E` is separable over `F`, then it's also separable
over `K`. -/
@[stacks 09H2 "second part"]
theorem Algebra.isSeparable_tower_top_of_isSeparable [Algebra.IsSeparable F E] :
    Algebra.IsSeparable L E :=
  ⟨fun x ↦ IsSeparable.tower_top _ (Algebra.IsSeparable.isSeparable F x)⟩


@[deprecated (since := "2024-08-06")]
alias IsSeparable.of_isScalarTower := Algebra.isSeparable_tower_top_of_isSeparable


variable {F} in
theorem isSeparable_algebraMap (x : F) : IsSeparable F (algebraMap F K x) :=
  Polynomial.Separable.of_dvd (Polynomial.separable_X_sub_C (x := x))
                                          /-
                                            F : Type u_1
                                            inst✝² : Field F
                                            K : Type u_2
                                            inst✝¹ : Ring K
                                            inst✝ : Algebra F K
                                            x : F
                                            ⊢ Eq ((Polynomial.aeval ((algebraMap F K) x)) (HSub.hSub Polynomial.X (Polynom …
                                          -/
    (minpoly.dvd F (algebraMap F K x) (by simp only [map_sub, aeval_X, aeval_C, sub_self]))
                                          /-
                                            🎉 no goals
                                          -/


instance Algebra.isSeparable_self : Algebra.IsSeparable F F :=
  ⟨isSeparable_algebraMap⟩


theorem IsSeparable.of_integral (x : K) : IsSeparable F x :=
  (minpoly.irreducible <| Algebra.IsIntegral.isIntegral x).separable

-- See note [lower instance priority]

variable (K) in
/-- A integral field extension in characteristic 0 is separable. -/
protected instance (priority := 100) Algebra.IsSeparable.of_integral : Algebra.IsSeparable F K :=
  ⟨_root_.IsSeparable.of_integral _⟩


variable {F} in
/-- If `E / K / F` is a scalar tower and `algebraMap K E x` is separable over `F`, then `x` is
also separable over `F`. -/
theorem IsSeparable.tower_bot {x : K} (h : IsSeparable F (algebraMap K E x)) : IsSeparable F x :=
    have ⟨_q, hq⟩ :=
      minpoly.dvd F x
        ((aeval_algebraMap_eq_zero_iff _ _ _).mp (minpoly.aeval F ((algebraMap K E) x)))
    (Eq.mp (congrArg Separable hq) h).of_mul_left


variable (K E) in
theorem Algebra.isSeparable_tower_bot_of_isSeparable [h : Algebra.IsSeparable F E] :
    Algebra.IsSeparable F K :=
  ⟨fun _ ↦ IsSeparable.tower_bot (h.isSeparable _ _)⟩


variable {F} in
theorem IsSeparable.of_algHom {x : E} (h : IsSeparable F (f x)) : IsSeparable F x := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    E' : Type u_4
    inst✝³ : Field E
    inst✝² : Field E'
    inst✝¹ : Algebra F E
    inst✝ : Algebra F E'
    f : AlgHom F E E'
    x : E
    h : IsSeparable F (f x)
    ⊢ IsSeparable F x
  -/
  let _ : Algebra E E' := RingHom.toAlgebra f.toRingHom
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    E' : Type u_4
    inst✝³ : Field E
    inst✝² : Field E'
    inst✝¹ : Algebra F E
    inst✝ : Algebra F E'
    f : AlgHom F E E'
    x : E
    h : IsSeparable F (f x)
    x✝ : Algebra E E' := f.toAlgebra
    ⊢ IsSeparable F x
  -/
  haveI : IsScalarTower F E E' := IsScalarTower.of_algebraMap_eq fun x => (f.commutes x).symm
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    E' : Type u_4
    inst✝³ : Field E
    inst✝² : Field E'
    inst✝¹ : Algebra F E
    inst✝ : Algebra F E'
    f : AlgHom F E E'
    x : E
    h : IsSeparable F (f x)
    x✝ : Algebra E E' := f.toAlgebra
    this : IsScalarTower F E E'
    ⊢ IsSeparable F x
  -/
  exact h.tower_bot
  /-
    🎉 no goals
  -/



variable (E') in
theorem Algebra.IsSeparable.of_algHom [Algebra.IsSeparable F E'] : Algebra.IsSeparable F E :=
  ⟨fun x => (Algebra.IsSeparable.isSeparable F (f x)).of_algHom⟩


instance isSeparable_tower_bot [Algebra.IsSeparable F K] : Algebra.IsSeparable F M :=
  Algebra.isSeparable_tower_bot_of_isSeparable F M K


instance isSeparable_tower_top [Algebra.IsSeparable F K] : Algebra.IsSeparable M K :=
  Algebra.isSeparable_tower_top_of_isSeparable F M K


lemma IsSeparable.of_equiv_equiv {x : B₁} (h : IsSeparable A₁ x) : IsSeparable A₂ (e₂ x) :=
  letI := e₁.toRingHom.toAlgebra
  letI : Algebra A₂ B₁ :=
    { (algebraMap A₁ B₁).comp e₁.symm.toRingHom with
        smul := fun a b ↦ ((algebraMap A₁ B₁).comp e₁.symm.toRingHom a) * b
        commutes' := fun r x ↦ (Algebra.commutes) (e₁.symm.toRingHom r) x
        smul_def' := fun _ _ ↦ rfl }
  haveI : IsScalarTower A₁ A₂ B₁ := IsScalarTower.of_algebraMap_eq <| fun x ↦
      (algebraMap A₁ B₁).congr_arg <| id ((e₁.symm_apply_apply x).symm)
  let e : B₁ ≃ₐ[A₂] B₂ :=
    { e₂ with
      commutes' := fun x ↦ by
        /-
          A₁ : Type u_1
          B₁ : Type u_2
          A₂ : Type u_3
          B₂ : Type u_4
          inst✝⁵ : Field A₁
          inst✝⁴ : Ring B₁
          inst✝³ : Field A₂
          inst✝² : Ring B₂
          inst✝¹ : Algebra A₁ B₁
          inst✝ : Algebra A₂ B₂
          e₁ : RingEquiv A₁ A₂
          e₂ : RingEquiv B₁ B₂
          he : Eq ((algebraMap A₂ B₂).comp ↑e₁) ((↑e₂).comp (algebraMap A₁ B₁))
          x✝ : B₁
          h : IsSeparable A₁ x✝
          this✝¹ : Algebra A₁ A₂ := e₁.toRingHom.toAlgebra
          this✝ : Algebra A₂ B₁ :=
            let __src := (algebraMap A₁ B₁).comp e₁.symm.toRingHom;
            Algebra.mk __src ⋯ ⋯
          this : IsScalarTower A₁ A₂ B₁
          x : A₂
          ⊢ Eq (e₂.toFun ((algebraMap A₂ B₁) x)) ((algebraMap A₂ B₂) x)
        -/
        simpa [RingHom.algebraMap_toAlgebra] using DFunLike.congr_fun he.symm (e₁.symm x) }
        /-
          🎉 no goals
        -/
  (AlgEquiv.isSeparable_iff e).mpr <| IsSeparable.tower_top A₂ h


lemma Algebra.IsSeparable.of_equiv_equiv [Algebra.IsSeparable A₁ B₁] : Algebra.IsSeparable A₂ B₂ :=
  ⟨fun x ↦ (e₂.apply_symm_apply x) ▸ _root_.IsSeparable.of_equiv_equiv e₁ e₂ he
    (Algebra.IsSeparable.isSeparable _ _)⟩


theorem AlgHom.card_of_powerBasis (pb : PowerBasis K S) (h_sep : IsSeparable K pb.gen)
    (h_splits : (minpoly K pb.gen).Splits (algebraMap K L)) :
    @Fintype.card (S →ₐ[K] L) (PowerBasis.AlgHom.fintype pb) = pb.dim := by
  classical
  let _ := (PowerBasis.AlgHom.fintype pb : Fintype (S →ₐ[K] L))
  rw [Fintype.card_congr pb.liftEquiv', Fintype.card_of_subtype _ (fun x => Multiset.mem_toFinset),
    ← pb.natDegree_minpoly, natDegree_eq_card_roots h_splits, Multiset.toFinset_card_of_nodup]
  exact nodup_roots ((separable_map (algebraMap K L)).mpr h_sep)


