/-- An `AddTorsor G P` gives a structure to the nonempty type `P`,
acted on by an `AddGroup G` with a transitive and free action given
by the `+ᵥ` operation and a corresponding subtraction given by the
`-ᵥ` operation. In the case of a vector space, it is an affine
space. -/
class AddTorsor (G : outParam Type*) (P : Type*) [AddGroup G] extends AddAction G P,
  VSub G P where
  [nonempty : Nonempty P]
  /-- Torsor subtraction and addition with the same element cancels out. -/
  vsub_vadd' : ∀ p₁ p₂ : P, (p₁ -ᵥ p₂ : G) +ᵥ p₂ = p₁
  /-- Torsor addition and subtraction with the same element cancels out. -/
  vadd_vsub' : ∀ (g : G) (p : P), (g +ᵥ p) -ᵥ p = g

 -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12096): removed `nolint instance_priority`; lint not ported yet

/-- An `AddGroup G` is a torsor for itself. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/12096): linter not ported yet
--@[nolint instance_priority]
instance addGroupIsAddTorsor (G : Type*) [AddGroup G] : AddTorsor G G where
  vsub := Sub.sub
  vsub_vadd' := sub_add_cancel
  vadd_vsub' := add_sub_cancel_right


/-- Simplify subtraction for a torsor for an `AddGroup G` over
itself. -/
@[simp]
theorem vsub_eq_sub {G : Type*} [AddGroup G] (g₁ g₂ : G) : g₁ -ᵥ g₂ = g₁ - g₂ :=
  rfl


/-- Adding the result of subtracting from another point produces that
point. -/
@[simp]
theorem vsub_vadd (p₁ p₂ : P) : (p₁ -ᵥ p₂) +ᵥ p₂ = p₁ :=
  AddTorsor.vsub_vadd' p₁ p₂


/-- Adding a group element then subtracting the original point
produces that group element. -/
@[simp]
theorem vadd_vsub (g : G) (p : P) : (g +ᵥ p) -ᵥ p = g :=
  AddTorsor.vadd_vsub' g p


/-- If the same point added to two group elements produces equal
results, those group elements are equal. -/
theorem vadd_right_cancel {g₁ g₂ : G} (p : P) (h : g₁ +ᵥ p = g₂ +ᵥ p) : g₁ = g₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    g₁ g₂ : G
    p : P
    h : Eq (HVAdd.hVAdd g₁ p) (HVAdd.hVAdd g₂ p)
    ⊢ Eq g₁ g₂
  -/
  rw [← vadd_vsub g₁ p, h, vadd_vsub]
  /-
    🎉 no goals
  -/


@[simp]
theorem vadd_right_cancel_iff {g₁ g₂ : G} (p : P) : g₁ +ᵥ p = g₂ +ᵥ p ↔ g₁ = g₂ :=
  ⟨vadd_right_cancel p, fun h => h ▸ rfl⟩


/-- Adding a group element to the point `p` is an injective
function. -/
theorem vadd_right_injective (p : P) : Function.Injective ((· +ᵥ p) : G → P) := fun _ _ =>
  vadd_right_cancel p


/-- Adding a group element to a point, then subtracting another point,
produces the same result as subtracting the points then adding the
group element. -/
theorem vadd_vsub_assoc (g : G) (p₁ p₂ : P) : (g +ᵥ p₁) -ᵥ p₂ = g + (p₁ -ᵥ p₂) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    g : G
    p₁ p₂ : P
    ⊢ Eq (VSub.vsub (HVAdd.hVAdd g p₁) p₂) (HAdd.hAdd g (VSub.vsub p₁ p₂))
  -/
  apply vadd_right_cancel p₂
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    g : G
    p₁ p₂ : P
    ⊢ Eq (HVAdd.hVAdd (VSub.vsub (HVAdd.hVAdd g p₁) p₂) p₂) (HVAdd.hVAdd (HAdd.hAd …
  -/
  rw [vsub_vadd, add_vadd, vsub_vadd]
  /-
    🎉 no goals
  -/


/-- Subtracting a point from itself produces 0. -/
@[simp]
theorem vsub_self (p : P) : p -ᵥ p = (0 : G) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p : P
    ⊢ Eq (VSub.vsub p p) 0
  -/
  rw [← zero_add (p -ᵥ p), ← vadd_vsub_assoc, vadd_vsub]
  /-
    🎉 no goals
  -/


/-- If subtracting two points produces 0, they are equal. -/
theorem eq_of_vsub_eq_zero {p₁ p₂ : P} (h : p₁ -ᵥ p₂ = (0 : G)) : p₁ = p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ : P
    h : Eq (VSub.vsub p₁ p₂) 0
    ⊢ Eq p₁ p₂
  -/
  rw [← vsub_vadd p₁ p₂, h, zero_vadd]
  /-
    🎉 no goals
  -/


/-- Subtracting two points produces 0 if and only if they are
equal. -/
@[simp]
theorem vsub_eq_zero_iff_eq {p₁ p₂ : P} : p₁ -ᵥ p₂ = (0 : G) ↔ p₁ = p₂ :=
  Iff.intro eq_of_vsub_eq_zero fun h => h ▸ vsub_self _


theorem vsub_ne_zero {p q : P} : p -ᵥ q ≠ (0 : G) ↔ p ≠ q :=
  not_congr vsub_eq_zero_iff_eq


/-- Cancellation adding the results of two subtractions. -/
@[simp]
theorem vsub_add_vsub_cancel (p₁ p₂ p₃ : P) : p₁ -ᵥ p₂ + (p₂ -ᵥ p₃) = p₁ -ᵥ p₃ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HAdd.hAdd (VSub.vsub p₁ p₂) (VSub.vsub p₂ p₃)) (VSub.vsub p₁ p₃)
  -/
  apply vadd_right_cancel p₃
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HVAdd.hVAdd (HAdd.hAdd (VSub.vsub p₁ p₂) (VSub.vsub p₂ p₃)) p₃) (HVAdd.h …
  -/
  rw [add_vadd, vsub_vadd, vsub_vadd, vsub_vadd]
  /-
    🎉 no goals
  -/


/-- Subtracting two points in the reverse order produces the negation
of subtracting them. -/
@[simp]
theorem neg_vsub_eq_vsub_rev (p₁ p₂ : P) : -(p₁ -ᵥ p₂) = p₂ -ᵥ p₁ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ : P
    ⊢ Eq (Neg.neg (VSub.vsub p₁ p₂)) (VSub.vsub p₂ p₁)
  -/
  refine neg_eq_of_add_eq_zero_right (vadd_right_cancel p₁ ?_)
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ : P
    ⊢ Eq (HVAdd.hVAdd (HAdd.hAdd (VSub.vsub p₁ p₂) (VSub.vsub p₂ p₁)) p₁) (HVAdd.h …
  -/
  rw [vsub_add_vsub_cancel, vsub_self]
  /-
    🎉 no goals
  -/


theorem vadd_vsub_eq_sub_vsub (g : G) (p q : P) : (g +ᵥ p) -ᵥ q = g - (q -ᵥ p) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    g : G
    p q : P
    ⊢ Eq (VSub.vsub (HVAdd.hVAdd g p) q) (HSub.hSub g (VSub.vsub q p))
  -/
  rw [vadd_vsub_assoc, sub_eq_add_neg, neg_vsub_eq_vsub_rev]
  /-
    🎉 no goals
  -/


/-- Subtracting the result of adding a group element produces the same result
as subtracting the points and subtracting that group element. -/
theorem vsub_vadd_eq_vsub_sub (p₁ p₂ : P) (g : G) : p₁ -ᵥ (g +ᵥ p₂) = p₁ -ᵥ p₂ - g := by
  rw [← add_right_inj (p₂ -ᵥ p₁ : G), vsub_add_vsub_cancel, ← neg_vsub_eq_vsub_rev, vadd_vsub, ←
    add_sub_assoc, ← neg_vsub_eq_vsub_rev, neg_add_cancel, zero_sub]


/-- Cancellation subtracting the results of two subtractions. -/
@[simp]
theorem vsub_sub_vsub_cancel_right (p₁ p₂ p₃ : P) : p₁ -ᵥ p₃ - (p₂ -ᵥ p₃) = p₁ -ᵥ p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HSub.hSub (VSub.vsub p₁ p₃) (VSub.vsub p₂ p₃)) (VSub.vsub p₁ p₂)
  -/
  rw [← vsub_vadd_eq_vsub_sub, vsub_vadd]
  /-
    🎉 no goals
  -/


/-- Convert between an equality with adding a group element to a point
and an equality of a subtraction of two points with a group
element. -/
theorem eq_vadd_iff_vsub_eq (p₁ : P) (g : G) (p₂ : P) : p₁ = g +ᵥ p₂ ↔ p₁ -ᵥ p₂ = g :=
  ⟨fun h => h.symm ▸ vadd_vsub _ _, fun h => h ▸ (vsub_vadd _ _).symm⟩


theorem vadd_eq_vadd_iff_neg_add_eq_vsub {v₁ v₂ : G} {p₁ p₂ : P} :
    v₁ +ᵥ p₁ = v₂ +ᵥ p₂ ↔ -v₁ + v₂ = p₁ -ᵥ p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    v₁ v₂ : G
    p₁ p₂ : P
    ⊢ Iff (Eq (HVAdd.hVAdd v₁ p₁) (HVAdd.hVAdd v₂ p₂)) (Eq (HAdd.hAdd (Neg.neg v₁) …
  -/
  rw [eq_vadd_iff_vsub_eq, vadd_vsub_assoc, ← add_right_inj (-v₁), neg_add_cancel_left, eq_comm]
  /-
    🎉 no goals
  -/


theorem singleton_vsub_self (p : P) : ({p} : Set P) -ᵥ {p} = {(0 : G)} := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p : P
    ⊢ Eq (VSub.vsub (Singleton.singleton p) (Singleton.singleton p)) (Singleton.si …
  -/
  rw [Set.singleton_vsub_singleton, vsub_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem vadd_vsub_vadd_cancel_right (v₁ v₂ : G) (p : P) : (v₁ +ᵥ p) -ᵥ (v₂ +ᵥ p) = v₁ - v₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    v₁ v₂ : G
    p : P
    ⊢ Eq (VSub.vsub (HVAdd.hVAdd v₁ p) (HVAdd.hVAdd v₂ p)) (HSub.hSub v₁ v₂)
  -/
  rw [vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, vsub_self, add_zero]
  /-
    🎉 no goals
  -/


/-- If the same point subtracted from two points produces equal
results, those points are equal. -/
theorem vsub_left_cancel {p₁ p₂ p : P} (h : p₁ -ᵥ p = p₂ -ᵥ p) : p₁ = p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p : P
    h : Eq (VSub.vsub p₁ p) (VSub.vsub p₂ p)
    ⊢ Eq p₁ p₂
  -/
  rwa [← sub_eq_zero, vsub_sub_vsub_cancel_right, vsub_eq_zero_iff_eq] at h
  /-
    🎉 no goals
  -/


/-- The same point subtracted from two points produces equal results
if and only if those points are equal. -/
@[simp]
theorem vsub_left_cancel_iff {p₁ p₂ p : P} : p₁ -ᵥ p = p₂ -ᵥ p ↔ p₁ = p₂ :=
  ⟨vsub_left_cancel, fun h => h ▸ rfl⟩


/-- Subtracting the point `p` is an injective function. -/
theorem vsub_left_injective (p : P) : Function.Injective ((· -ᵥ p) : P → G) := fun _ _ =>
  vsub_left_cancel


/-- If subtracting two points from the same point produces equal
results, those points are equal. -/
theorem vsub_right_cancel {p₁ p₂ p : P} (h : p -ᵥ p₁ = p -ᵥ p₂) : p₁ = p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p : P
    h : Eq (VSub.vsub p p₁) (VSub.vsub p p₂)
    ⊢ Eq p₁ p₂
  -/
  refine vadd_left_cancel (p -ᵥ p₂) ?_
  /-
    G : Type u_1
    P : Type u_2
    inst✝ : AddGroup G
    T : AddTorsor G P
    p₁ p₂ p : P
    h : Eq (VSub.vsub p p₁) (VSub.vsub p p₂)
    ⊢ Eq (HVAdd.hVAdd (VSub.vsub p p₂) p₁) (HVAdd.hVAdd (VSub.vsub p p₂) p₂)
  -/
  rw [vsub_vadd, ← h, vsub_vadd]
  /-
    🎉 no goals
  -/


/-- Subtracting two points from the same point produces equal results
if and only if those points are equal. -/
@[simp]
theorem vsub_right_cancel_iff {p₁ p₂ p : P} : p -ᵥ p₁ = p -ᵥ p₂ ↔ p₁ = p₂ :=
  ⟨vsub_right_cancel, fun h => h ▸ rfl⟩


/-- Subtracting a point from the point `p` is an injective
function. -/
theorem vsub_right_injective (p : P) : Function.Injective ((p -ᵥ ·) : P → G) := fun _ _ =>
  vsub_right_cancel


/-- Cancellation subtracting the results of two subtractions. -/
@[simp]
theorem vsub_sub_vsub_cancel_left (p₁ p₂ p₃ : P) : p₃ -ᵥ p₂ - (p₃ -ᵥ p₁) = p₁ -ᵥ p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HSub.hSub (VSub.vsub p₃ p₂) (VSub.vsub p₃ p₁)) (VSub.vsub p₁ p₂)
  -/
  rw [sub_eq_add_neg, neg_vsub_eq_vsub_rev, add_comm, vsub_add_vsub_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem vadd_vsub_vadd_cancel_left (v : G) (p₁ p₂ : P) : (v +ᵥ p₁) -ᵥ (v +ᵥ p₂) = p₁ -ᵥ p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    v : G
    p₁ p₂ : P
    ⊢ Eq (VSub.vsub (HVAdd.hVAdd v p₁) (HVAdd.hVAdd v p₂)) (VSub.vsub p₁ p₂)
  -/
  rw [vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


theorem vsub_vadd_comm (p₁ p₂ p₃ : P) : (p₁ -ᵥ p₂ : G) +ᵥ p₃ = (p₃ -ᵥ p₂) +ᵥ p₁ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HVAdd.hVAdd (VSub.vsub p₁ p₂) p₃) (HVAdd.hVAdd (VSub.vsub p₃ p₂) p₁)
  -/
  rw [← @vsub_eq_zero_iff_eq G, vadd_vsub_assoc, vsub_vadd_eq_vsub_sub]
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    p₁ p₂ p₃ : P
    ⊢ Eq (HAdd.hAdd (VSub.vsub p₁ p₂) (HSub.hSub (VSub.vsub p₃ p₁) (VSub.vsub p₃ p …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem vadd_eq_vadd_iff_sub_eq_vsub {v₁ v₂ : G} {p₁ p₂ : P} :
    v₁ +ᵥ p₁ = v₂ +ᵥ p₂ ↔ v₂ - v₁ = p₁ -ᵥ p₂ := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    v₁ v₂ : G
    p₁ p₂ : P
    ⊢ Iff (Eq (HVAdd.hVAdd v₁ p₁) (HVAdd.hVAdd v₂ p₂)) (Eq (HSub.hSub v₂ v₁) (VSub …
  -/
  rw [vadd_eq_vadd_iff_neg_add_eq_vsub, neg_add_eq_sub]
  /-
    🎉 no goals
  -/


theorem vsub_sub_vsub_comm (p₁ p₂ p₃ p₄ : P) : p₁ -ᵥ p₂ - (p₃ -ᵥ p₄) = p₁ -ᵥ p₃ - (p₂ -ᵥ p₄) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : AddTorsor G P
    p₁ p₂ p₃ p₄ : P
    ⊢ Eq (HSub.hSub (VSub.vsub p₁ p₂) (VSub.vsub p₃ p₄)) (HSub.hSub (VSub.vsub p₁  …
  -/
  rw [← vsub_vadd_eq_vsub_sub, vsub_vadd_comm, vsub_vadd_eq_vsub_sub]
  /-
    🎉 no goals
  -/


instance instAddTorsor : AddTorsor (G × G') (P × P') where
  vadd v p := (v.1 +ᵥ p.1, v.2 +ᵥ p.2)
  zero_vadd _ := Prod.ext (zero_vadd _ _) (zero_vadd _ _)
  add_vadd _ _ _ := Prod.ext (add_vadd _ _ _) (add_vadd _ _ _)
  vsub p₁ p₂ := (p₁.1 -ᵥ p₂.1, p₁.2 -ᵥ p₂.2)
  vsub_vadd' _ _ := Prod.ext (vsub_vadd _ _) (vsub_vadd _ _)
  vadd_vsub' _ _ := Prod.ext (vadd_vsub _ _) (vadd_vsub _ _)

-- Porting note: The proofs above used to be shorter:
-- zero_vadd p := by simp ⊢ 0 +ᵥ p = p
-- add_vadd := by simp [add_vadd] ⊢ ∀ (a : G) (b : G') (a_1 : G) (b_1 : G') (a_2 : P) (b_2 : P'),
--  (a + a_1, b + b_1) +ᵥ (a_2, b_2) = (a, b) +ᵥ ((a_1, b_1) +ᵥ (a_2, b_2))
-- vsub_vadd' p₁ p₂ := show (p₁.1 -ᵥ p₂.1 +ᵥ p₂.1, _) = p₁ by simp
--   ⊢ (p₁.fst -ᵥ p₂.fst +ᵥ p₂.fst, ((p₁.fst -ᵥ p₂.fst, p₁.snd -ᵥ p₂.snd) +ᵥ p₂).snd) = p₁
-- vadd_vsub' v p := show (v.1 +ᵥ p.1 -ᵥ p.1, v.2 +ᵥ p.2 -ᵥ p.2) = v by simp
--   ⊢ (v.fst +ᵥ p.fst -ᵥ p.fst, v.snd) = v


@[simp]
theorem fst_vadd (v : G × G') (p : P × P') : (v +ᵥ p).1 = v.1 +ᵥ p.1 :=
  rfl


@[simp]
theorem snd_vadd (v : G × G') (p : P × P') : (v +ᵥ p).2 = v.2 +ᵥ p.2 :=
  rfl


@[simp]
theorem mk_vadd_mk (v : G) (v' : G') (p : P) (p' : P') : (v, v') +ᵥ (p, p') = (v +ᵥ p, v' +ᵥ p') :=
  rfl


@[simp]
theorem fst_vsub (p₁ p₂ : P × P') : (p₁ -ᵥ p₂ : G × G').1 = p₁.1 -ᵥ p₂.1 :=
  rfl


@[simp]
theorem snd_vsub (p₁ p₂ : P × P') : (p₁ -ᵥ p₂ : G × G').2 = p₁.2 -ᵥ p₂.2 :=
  rfl


@[simp]
theorem mk_vsub_mk (p₁ p₂ : P) (p₁' p₂' : P') :
    ((p₁, p₁') -ᵥ (p₂, p₂') : G × G') = (p₁ -ᵥ p₂, p₁' -ᵥ p₂') :=
  rfl


/-- A product of `AddTorsor`s is an `AddTorsor`. -/
instance instAddTorsor [∀ i, AddTorsor (fg i) (fp i)] : AddTorsor (∀ i, fg i) (∀ i, fp i) where
  vadd g p i := g i +ᵥ p i
  zero_vadd p := funext fun i => zero_vadd (fg i) (p i)
  add_vadd g₁ g₂ p := funext fun i => add_vadd (g₁ i) (g₂ i) (p i)
  vsub p₁ p₂ i := p₁ i -ᵥ p₂ i
  vsub_vadd' p₁ p₂ := funext fun i => vsub_vadd (p₁ i) (p₂ i)
  vadd_vsub' g p := funext fun i => vadd_vsub (g i) (p i)


/-- `v ↦ v +ᵥ p` as an equivalence. -/
def vaddConst (p : P) : G ≃ P where
  toFun v := v +ᵥ p
  invFun p' := p' -ᵥ p
  left_inv _ := vadd_vsub _ _
  right_inv _ := vsub_vadd _ _


@[simp]
theorem coe_vaddConst (p : P) : ⇑(vaddConst p) = fun v => v +ᵥ p :=
  rfl


@[simp]
theorem coe_vaddConst_symm (p : P) : ⇑(vaddConst p).symm = fun p' => p' -ᵥ p :=
  rfl


/-- `p' ↦ p -ᵥ p'` as an equivalence. -/
def constVSub (p : P) : P ≃ G where
  toFun := (p -ᵥ ·)
  invFun := (-· +ᵥ p)
                    /-
                      G : Type u_1
                      P : Type u_2
                      inst✝¹ : AddGroup G
                      inst✝ : AddTorsor G P
                      p p' : P
                      ⊢ Eq ((fun x => HVAdd.hVAdd (Neg.neg x) p) ((fun x => VSub.vsub p x) p')) p'
                    -/
  left_inv p' := by simp
                    /-
                      🎉 no goals
                    -/
                    /-
                      G : Type u_1
                      P : Type u_2
                      inst✝¹ : AddGroup G
                      inst✝ : AddTorsor G P
                      p : P
                      v : G
                      ⊢ Eq ((fun x => VSub.vsub p x) ((fun x => HVAdd.hVAdd (Neg.neg x) p) v)) v
                    -/
  right_inv v := by simp [vsub_vadd_eq_vsub_sub]
                    /-
                      🎉 no goals
                    -/


@[simp] lemma coe_constVSub (p : P) : ⇑(constVSub p) = (p -ᵥ ·) := rfl


@[simp]
theorem coe_constVSub_symm (p : P) : ⇑(constVSub p).symm = fun (v : G) => -v +ᵥ p :=
  rfl


/-- The permutation given by `p ↦ v +ᵥ p`. -/
def constVAdd (v : G) : Equiv.Perm P where
  toFun := (v +ᵥ ·)
  invFun := (-v +ᵥ ·)
                   /-
                     G : Type u_1
                     P : Type u_2
                     inst✝¹ : AddGroup G
                     inst✝ : AddTorsor G P
                     v : G
                     p : P
                     ⊢ Eq ((fun x => HVAdd.hVAdd (Neg.neg v) x) ((fun x => HVAdd.hVAdd v x) p)) p
                   -/
  left_inv p := by simp [vadd_vadd]
                   /-
                     🎉 no goals
                   -/
                    /-
                      G : Type u_1
                      P : Type u_2
                      inst✝¹ : AddGroup G
                      inst✝ : AddTorsor G P
                      v : G
                      p : P
                      ⊢ Eq ((fun x => HVAdd.hVAdd v x) ((fun x => HVAdd.hVAdd (Neg.neg v) x) p)) p
                    -/
  right_inv p := by simp [vadd_vadd]
                    /-
                      🎉 no goals
                    -/


@[simp] lemma coe_constVAdd (v : G) : ⇑(constVAdd P v) = (v +ᵥ ·) := rfl


@[simp]
theorem constVAdd_zero : constVAdd P (0 : G) = 1 :=
  ext <| zero_vadd G


@[simp]
theorem constVAdd_add (v₁ v₂ : G) : constVAdd P (v₁ + v₂) = constVAdd P v₁ * constVAdd P v₂ :=
  ext <| add_vadd v₁ v₂


/-- `Equiv.constVAdd` as a homomorphism from `Multiplicative G` to `Equiv.perm P` -/
def constVAddHom : Multiplicative G →* Equiv.Perm P where
  toFun v := constVAdd P (v.toAdd)
  map_one' := constVAdd_zero G P
  map_mul' := constVAdd_add P


/-- Point reflection in `x` as a permutation. -/
def pointReflection (x : P) : Perm P :=
  (constVSub x).trans (vaddConst x)


theorem pointReflection_apply (x y : P) : pointReflection x y = (x -ᵥ y) +ᵥ x :=
  rfl


@[simp]
theorem pointReflection_vsub_left (x y : P) : pointReflection x y -ᵥ x = x -ᵥ y :=
  vadd_vsub ..


@[simp]
theorem left_vsub_pointReflection (x y : P) : x -ᵥ pointReflection x y = y -ᵥ x :=
                      /-
                        G : Type u_1
                        P : Type u_2
                        inst✝¹ : AddGroup G
                        inst✝ : AddTorsor G P
                        x y : P
                        ⊢ Eq (Neg.neg (VSub.vsub x ((Equiv.pointReflection x) y))) (Neg.neg (VSub.vsub …
                      -/
  neg_injective <| by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem pointReflection_vsub_right (x y : P) : pointReflection x y -ᵥ y = 2 • (x -ᵥ y) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddGroup G
    inst✝ : AddTorsor G P
    x y : P
    ⊢ Eq (VSub.vsub ((Equiv.pointReflection x) y) y) (HSMul.hSMul 2 (VSub.vsub x y))
  -/
  simp [pointReflection, two_nsmul, vadd_vsub_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem right_vsub_pointReflection (x y : P) : y -ᵥ pointReflection x y = 2 • (y -ᵥ x) :=
                      /-
                        G : Type u_1
                        P : Type u_2
                        inst✝¹ : AddGroup G
                        inst✝ : AddTorsor G P
                        x y : P
                        ⊢ Eq (Neg.neg (VSub.vsub y ((Equiv.pointReflection x) y))) (Neg.neg (HSMul.hSM …
                      -/
  neg_injective <| by simp [← neg_nsmul]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem pointReflection_symm (x : P) : (pointReflection x).symm = pointReflection x :=
            /-
              G : Type u_1
              P : Type u_2
              inst✝¹ : AddGroup G
              inst✝ : AddTorsor G P
              x : P
              ⊢ ∀ (x_1 : P), Eq ((Equiv.symm (Equiv.pointReflection x)) x_1) ((Equiv.pointRe …
            -/
  ext <| by simp [pointReflection]
            /-
              🎉 no goals
            -/


@[simp]
theorem pointReflection_self (x : P) : pointReflection x x = x :=
  vsub_vadd _ _


theorem pointReflection_involutive (x : P) : Involutive (pointReflection x : P → P) := fun y =>
                                               /-
                                                 G : Type u_1
                                                 P : Type u_2
                                                 inst✝¹ : AddGroup G
                                                 inst✝ : AddTorsor G P
                                                 x y : P
                                                 ⊢ Eq ((Equiv.pointReflection x) y) ((Equiv.symm (Equiv.pointReflection x)) y)
                                               -/
  (Equiv.apply_eq_iff_eq_symm_apply _).2 <| by rw [pointReflection_symm]
                                               /-
                                                 🎉 no goals
                                               -/


/-- `x` is the only fixed point of `pointReflection x`. This lemma requires
`x + x = y + y ↔ x = y`. There is no typeclass to use here, so we add it as an explicit argument. -/
theorem pointReflection_fixed_iff_of_injective_two_nsmul {x y : P} (h : Injective (2 • · : G → G)) :
    pointReflection x y = y ↔ y = x := by
  rw [pointReflection_apply, eq_comm, eq_vadd_iff_vsub_eq, ← neg_vsub_eq_vsub_rev,
    neg_eq_iff_add_eq_zero, ← two_nsmul, ← nsmul_zero 2, h.eq_iff, vsub_eq_zero_iff_eq, eq_comm]


@[deprecated (since := "2024-11-18")] alias pointReflection_fixed_iff_of_injective_bit0 :=
pointReflection_fixed_iff_of_injective_two_nsmul


theorem injective_pointReflection_left_of_injective_two_nsmul {G P : Type*} [AddCommGroup G]
    [AddTorsor G P] (h : Injective (2 • · : G → G)) (y : P) :
    Injective fun x : P => pointReflection x y :=
  fun x₁ x₂ (hy : pointReflection x₁ y = pointReflection x₂ y) => by
  rwa [pointReflection_apply, pointReflection_apply, vadd_eq_vadd_iff_sub_eq_vsub,
    vsub_sub_vsub_cancel_right, ← neg_vsub_eq_vsub_rev, neg_eq_iff_add_eq_zero,
    ← two_nsmul, ← nsmul_zero 2, h.eq_iff, vsub_eq_zero_iff_eq] at hy


@[deprecated (since := "2024-11-18")] alias injective_pointReflection_left_of_injective_bit0 :=
injective_pointReflection_left_of_injective_two_nsmul


theorem AddTorsor.subsingleton_iff (G P : Type*) [AddGroup G] [AddTorsor G P] :
    Subsingleton G ↔ Subsingleton P := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddGroup G
    inst✝ : AddTorsor G P
    ⊢ Iff (Subsingleton G) (Subsingleton P)
  -/
  inhabit P
  /-
    G : Type u_1
    P : Type u_2
    inst✝¹ : AddGroup G
    inst✝ : AddTorsor G P
    inhabited_h : Inhabited P
    ⊢ Iff (Subsingleton G) (Subsingleton P)
  -/
  exact (Equiv.vaddConst default).subsingleton_congr
  /-
    🎉 no goals
  -/

