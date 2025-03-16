/--
A two-sided ideal of a ring `R` is a subset of `R` that contains `0` and is closed under addition,
negation, and absorbs multiplication on both sides.
-/
structure TwoSidedIdeal (R : Type*) [NonUnitalNonAssocRing R] where
  /-- every two-sided-ideal is induced by a congruence relation on the ring. -/
  ringCon : RingCon R


instance [Nontrivial R] : Nontrivial (TwoSidedIdeal R) := by
  /-
    R : Type u_1
    inst✝¹ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    inst✝ : Nontrivial R
    ⊢ Nontrivial (TwoSidedIdeal R)
  -/
  obtain ⟨I, J, h⟩ : Nontrivial (RingCon R) := inferInstance
  /-
    case mk.intro.intro
    R : Type u_1
    inst✝¹ : NonUnitalNonAssocRing R
    I✝ : TwoSidedIdeal R
    inst✝ : Nontrivial R
    I J : RingCon R
    h : Ne I J
    ⊢ Nontrivial (TwoSidedIdeal R)
  -/
  exact ⟨⟨I⟩, ⟨J⟩, by contrapose! h; aesop⟩
  /-
    🎉 no goals
  -/


instance setLike : SetLike (TwoSidedIdeal R) R where
  coe t := {r | t.ringCon r 0}
  coe_injective'  := by
    /-
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I : TwoSidedIdeal R
      ⊢ Function.Injective fun t => setOf fun r => t.ringCon r 0
    -/
    rintro ⟨t₁⟩ ⟨t₂⟩ (h : {x | _} = {x | _})
    /-
      case mk.mk
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I : TwoSidedIdeal R
      t₁ t₂ : RingCon R
      h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
      ⊢ Eq { ringCon := t₁ } { ringCon := t₂ }
    -/
    congr 1
    /-
      case mk.mk.e_ringCon
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I : TwoSidedIdeal R
      t₁ t₂ : RingCon R
      h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
      ⊢ Eq t₁ t₂
    -/
    refine RingCon.ext fun a b ↦ ⟨fun H ↦ ?_, fun H ↦ ?_⟩
      /-
        case mk.mk.e_ringCon.refine_1
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₁ a b
        ⊢ t₂ a b
      -/
    · have H' : a - b ∈ {x | t₁ x 0} := sub_self b ▸ t₁.sub H (t₁.refl b)
      /-
        case mk.mk.e_ringCon.refine_1
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₁ a b
        H' : Membership.mem (setOf fun x => t₁ x 0) (HSub.hSub a b)
        ⊢ t₂ a b
      -/
      rw [h] at H'
      /-
        case mk.mk.e_ringCon.refine_1
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₁ a b
        H' : Membership.mem (setOf fun x => { ringCon := t₂ }.ringCon x 0) (HSub.hSub  …
        ⊢ t₂ a b
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
      convert t₂.add H' (t₂.refl b) using 1 <;> abel
                                                /-
                                                  🎉 no goals
                                                -/
      /-
        case mk.mk.e_ringCon.refine_2
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₂ a b
        ⊢ t₁ a b
      -/
    · have H' : a - b ∈ {x | t₂ x 0} := sub_self b ▸ t₂.sub H (t₂.refl b)
      /-
        case mk.mk.e_ringCon.refine_2
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₂ a b
        H' : Membership.mem (setOf fun x => t₂ x 0) (HSub.hSub a b)
        ⊢ t₁ a b
      -/
      rw [← h] at H'
      /-
        case mk.mk.e_ringCon.refine_2
        R : Type u_1
        inst✝ : NonUnitalNonAssocRing R
        I : TwoSidedIdeal R
        t₁ t₂ : RingCon R
        h : Eq (setOf fun x => { ringCon := t₁ }.ringCon x 0) (setOf fun x => { ringCo …
        a b : R
        H : t₂ a b
        H' : Membership.mem (setOf fun x => { ringCon := t₁ }.ringCon x 0) (HSub.hSub  …
        ⊢ t₁ a b
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
      convert t₁.add H' (t₁.refl b) using 1 <;> abel
                                                /-
                                                  🎉 no goals
                                                -/


lemma mem_iff (x : R) : x ∈ I ↔ I.ringCon x 0 := Iff.rfl


lemma rel_iff (x y : R) : I.ringCon x y ↔ x - y ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    x y : R
    ⊢ Iff (I.ringCon x y) (Membership.mem I (HSub.hSub x y))
  -/
  rw [mem_iff]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    x y : R
    ⊢ Iff (I.ringCon x y) (I.ringCon (HSub.hSub x y) 0)
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I : TwoSidedIdeal R
      x y : R
      ⊢ I.ringCon x y → I.ringCon (HSub.hSub x y) 0
    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  · intro h; convert I.ringCon.sub h (I.ringCon.refl y); abel
                                                         /-
                                                           🎉 no goals
                                                         -/
    /-
      case mpr
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I : TwoSidedIdeal R
      x y : R
      ⊢ I.ringCon (HSub.hSub x y) 0 → I.ringCon x y
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
  · intro h; convert I.ringCon.add h (I.ringCon.refl y) <;> abel
                                                            /-
                                                              🎉 no goals
                                                            -/


/--
the coercion from two-sided-ideals to sets is an order embedding
-/
@[simps]
def coeOrderEmbedding : TwoSidedIdeal R ↪o Set R where
  toFun := SetLike.coe
  inj' := SetLike.coe_injective
  map_rel_iff' {I J} := ⟨fun (h : (I : Set R) ⊆ (J : Set R)) _ h' ↦ h h', fun h _ h' ↦ h h'⟩


lemma le_iff {I J : TwoSidedIdeal R} : I ≤ J ↔ (I : Set R) ⊆ (J : Set R) := Iff.rfl


/-- Two-sided-ideals corresponds to congruence relations on a ring. -/
def orderIsoRingCon : TwoSidedIdeal R ≃o RingCon R where
  toFun := TwoSidedIdeal.ringCon
  invFun := .mk
  left_inv _ := rfl
  right_inv _ := rfl
                                                                    /-
                                                                      R : Type u_1
                                                                      inst✝ : NonUnitalNonAssocRing R
                                                                      I✝ I J : TwoSidedIdeal R
                                                                      h : HasSubset.Subset ↑I ↑J
                                                                      x y : R
                                                                      r : ({ toFun := TwoSidedIdeal.ringCon, invFun := TwoSidedIdeal.mk, left_inv := …
                                                                      ⊢ ({ toFun := TwoSidedIdeal.ringCon, invFun := TwoSidedIdeal.mk, left_inv := ⋯ …
                                                                    -/
  map_rel_iff' {I J} := Iff.symm <| le_iff.trans ⟨fun h x y r => by rw [rel_iff] at r ⊢; exact h r,
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
                     /-
                       R : Type u_1
                       inst✝ : NonUnitalNonAssocRing R
                       I✝ I J : TwoSidedIdeal R
                       h : LE.le ({ toFun := TwoSidedIdeal.ringCon, invFun := TwoSidedIdeal.mk, left_ …
                       x : R
                       hx : Membership.mem (↑I) x
                       ⊢ Membership.mem (↑J) x
                     -/
    fun h x hx => by rw [SetLike.mem_coe, mem_iff] at hx ⊢; exact h hx⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma ringCon_injective : Function.Injective (TwoSidedIdeal.ringCon (R := R)) := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    ⊢ Function.Injective TwoSidedIdeal.ringCon
  -/
  rintro ⟨x⟩ ⟨y⟩ rfl; rfl
                      /-
                        🎉 no goals
                      -/


lemma ringCon_le_iff {I J : TwoSidedIdeal R} : I ≤ J ↔ I.ringCon ≤ J.ringCon :=
  orderIsoRingCon.map_rel_iff.symm


@[ext]
lemma ext {I J : TwoSidedIdeal R} (h : ∀ x, x ∈ I ↔ x ∈ J) : I = J :=
  coeOrderEmbedding.injective (Set.ext h)


lemma lt_iff (I J : TwoSidedIdeal R) : I < J ↔ (I : Set R) ⊂ (J : Set R) := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I J : TwoSidedIdeal R
    ⊢ Iff (LT.lt I J) (HasSSubset.SSubset ↑I ↑J)
  -/
  rw [lt_iff_le_and_ne, Set.ssubset_iff_subset_ne, le_iff]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I J : TwoSidedIdeal R
    ⊢ Iff (And (HasSubset.Subset ↑I ↑J) (Ne I J)) (And (HasSubset.Subset ↑I ↑J) (N …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma zero_mem : 0 ∈ I := I.ringCon.refl 0


                                                                /-
                                                                  R : Type u_1
                                                                  inst✝ : NonUnitalNonAssocRing R
                                                                  I : TwoSidedIdeal R
                                                                  x y : R
                                                                  hx : Membership.mem I x
                                                                  hy : Membership.mem I y
                                                                  ⊢ Membership.mem I (HAdd.hAdd x y)
                                                                -/
lemma add_mem {x y} (hx : x ∈ I) (hy : y ∈ I) : x + y ∈ I := by simpa using I.ringCon.add hx hy
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                              /-
                                                R : Type u_1
                                                inst✝ : NonUnitalNonAssocRing R
                                                I : TwoSidedIdeal R
                                                x : R
                                                hx : Membership.mem I x
                                                ⊢ Membership.mem I (Neg.neg x)
                                              -/
lemma neg_mem {x} (hx : x ∈ I) : -x ∈ I := by simpa using I.ringCon.neg hx
                                              /-
                                                🎉 no goals
                                              -/


instance : AddSubgroupClass (TwoSidedIdeal R) R where
  zero_mem := zero_mem
  add_mem := @add_mem _ _
  neg_mem := @neg_mem _ _


lemma sub_mem {x y} (hx : x ∈ I) (hy : y ∈ I) : x - y ∈ I := _root_.sub_mem hx hy


lemma mul_mem_left (x y) (hy : y ∈ I) : x * y ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    x y : R
    hy : Membership.mem I y
    ⊢ Membership.mem I (HMul.hMul x y)
  -/
  simpa using I.ringCon.mul (I.ringCon.refl x) hy
  /-
    🎉 no goals
  -/


lemma mul_mem_right (x y) (hx : x ∈ I) : x * y ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I : TwoSidedIdeal R
    x y : R
    hx : Membership.mem I x
    ⊢ Membership.mem I (HMul.hMul x y)
  -/
  simpa using I.ringCon.mul hx (I.ringCon.refl y)
  /-
    🎉 no goals
  -/


lemma nsmul_mem {x} (n : ℕ) (hx : x ∈ I) : n • x ∈ I := _root_.nsmul_mem hx _


lemma zsmul_mem {x} (n : ℤ) (hx : x ∈ I) : n • x ∈ I := _root_.zsmul_mem hx _


/--
The "set-theoretic-way" of constructing a two-sided ideal by providing:
- the underlying set `S`;
- a proof that `0 ∈ S`;
- a proof that `x + y ∈ S` if `x ∈ S` and `y ∈ S`;
- a proof that `-x ∈ S` if `x ∈ S`;
- a proof that `x * y ∈ S` if `y ∈ S`;
- a proof that `x * y ∈ S` if `x ∈ S`.
-/
def mk' (carrier : Set R)
    (zero_mem : 0 ∈ carrier)
    (add_mem : ∀ {x y}, x ∈ carrier → y ∈ carrier → x + y ∈ carrier)
    (neg_mem : ∀ {x}, x ∈ carrier → -x ∈ carrier)
    (mul_mem_left : ∀ {x y}, y ∈ carrier → x * y ∈ carrier)
    (mul_mem_right : ∀ {x y}, x ∈ carrier → x * y ∈ carrier) : TwoSidedIdeal R where
  ringCon :=
    { r := fun x y ↦ x - y ∈ carrier
      iseqv :=
                           /-
                             R : Type u_1
                             inst✝ : NonUnitalNonAssocRing R
                             I : TwoSidedIdeal R
                             carrier : Set R
                             zero_mem : Membership.mem carrier 0
                             add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
                             neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
                             mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
                             mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
                             x : R
                             ⊢ Membership.mem carrier (HSub.hSub x x)
                           -/
      { refl := fun x ↦ by simpa using zero_mem
                           /-
                             🎉 no goals
                           -/
                           /-
                             R : Type u_1
                             inst✝ : NonUnitalNonAssocRing R
                             I : TwoSidedIdeal R
                             carrier : Set R
                             zero_mem : Membership.mem carrier 0
                             add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
                             neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
                             mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
                             mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
                             x✝ y✝ : R
                             h : Membership.mem carrier (HSub.hSub x✝ y✝)
                             ⊢ Membership.mem carrier (HSub.hSub y✝ x✝)
                           -/
        symm := fun h ↦ by simpa using neg_mem h
                           /-
                             🎉 no goals
                           -/
        trans := fun {x y z} h1 h2 ↦ by
          /-
            R : Type u_1
            inst✝ : NonUnitalNonAssocRing R
            I : TwoSidedIdeal R
            carrier : Set R
            zero_mem : Membership.mem carrier 0
            add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
            neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
            mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
            mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
            x y z : R
            h1 : Membership.mem carrier (HSub.hSub x y)
            h2 : Membership.mem carrier (HSub.hSub y z)
            ⊢ Membership.mem carrier (HSub.hSub x z)
          -/
          simpa only [show x - z = (x - y) + (y - z) by abel] using add_mem h1 h2 }
          /-
            🎉 no goals
          -/
      mul' := fun {a b c d} (h1 : a - b ∈ carrier) (h2 : c - d ∈ carrier) ↦ show _ ∈ carrier by
        /-
          R : Type u_1
          inst✝ : NonUnitalNonAssocRing R
          I : TwoSidedIdeal R
          carrier : Set R
          zero_mem : Membership.mem carrier 0
          add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
          neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
          mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
          mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
          a b c d : R
          h1 : Membership.mem carrier (HSub.hSub a b)
          h2 : Membership.mem carrier (HSub.hSub c d)
          ⊢ Membership.mem carrier (HSub.hSub (HMul.hMul a c) (HMul.hMul b d))
        -/
        rw [show a * c - b * d = a * (c - d) + (a - b) * d by rw [mul_sub, sub_mul]; abel]
        /-
          R : Type u_1
          inst✝ : NonUnitalNonAssocRing R
          I : TwoSidedIdeal R
          carrier : Set R
          zero_mem : Membership.mem carrier 0
          add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
          neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
          mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
          mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
          a b c d : R
          h1 : Membership.mem carrier (HSub.hSub a b)
          h2 : Membership.mem carrier (HSub.hSub c d)
          ⊢ Membership.mem carrier (HAdd.hAdd (HMul.hMul a (HSub.hSub c d)) (HMul.hMul ( …
        -/
        exact add_mem (mul_mem_left h2) (mul_mem_right h1)
        /-
          🎉 no goals
        -/
      add' := fun {a b c d} (h1 : a - b ∈ carrier) (h2 : c - d ∈ carrier) ↦ show _ ∈ carrier by
        /-
          R : Type u_1
          inst✝ : NonUnitalNonAssocRing R
          I : TwoSidedIdeal R
          carrier : Set R
          zero_mem : Membership.mem carrier 0
          add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
          neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
          mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
          mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
          a b c d : R
          h1 : Membership.mem carrier (HSub.hSub a b)
          h2 : Membership.mem carrier (HSub.hSub c d)
          ⊢ Membership.mem carrier (HSub.hSub (HAdd.hAdd a c) (HAdd.hAdd b d))
        -/
        rw [show a + c - (b + d) = (a - b) + (c - d) by abel]
        /-
          R : Type u_1
          inst✝ : NonUnitalNonAssocRing R
          I : TwoSidedIdeal R
          carrier : Set R
          zero_mem : Membership.mem carrier 0
          add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
          neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
          mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
          mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
          a b c d : R
          h1 : Membership.mem carrier (HSub.hSub a b)
          h2 : Membership.mem carrier (HSub.hSub c d)
          ⊢ Membership.mem carrier (HAdd.hAdd (HSub.hSub a b) (HSub.hSub c d))
        -/
        exact add_mem h1 h2 }
        /-
          🎉 no goals
        -/


@[simp]
lemma mem_mk' (carrier : Set R) (zero_mem add_mem neg_mem mul_mem_left mul_mem_right) (x : R) :
    x ∈ mk' carrier zero_mem add_mem neg_mem mul_mem_left mul_mem_right ↔ x ∈ carrier := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    carrier : Set R
    zero_mem : Membership.mem carrier 0
    add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
    neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
    mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
    mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
    x : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.mk' carrier zero_mem add_mem neg_mem mul_ …
  -/
  rw [mem_iff]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    carrier : Set R
    zero_mem : Membership.mem carrier 0
    add_mem : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier y → M …
    neg_mem : ∀ {x : R}, Membership.mem carrier x → Membership.mem carrier (Neg.ne …
    mul_mem_left : ∀ {x y : R}, Membership.mem carrier y → Membership.mem carrier  …
    mul_mem_right : ∀ {x y : R}, Membership.mem carrier x → Membership.mem carrier …
    x : R
    ⊢ Iff ((TwoSidedIdeal.mk' carrier zero_mem add_mem neg_mem mul_mem_left mul_me …
  -/
  simp [mk']
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
@[simp]
lemma coe_mk' (carrier : Set R) (zero_mem add_mem neg_mem mul_mem_left mul_mem_right) :
    (mk' carrier zero_mem add_mem neg_mem mul_mem_left mul_mem_right : Set R) = carrier :=
  Set.ext <| mem_mk' carrier zero_mem add_mem neg_mem mul_mem_left mul_mem_right


instance : SMulMemClass (TwoSidedIdeal R) R R where
  smul_mem _ _ h := TwoSidedIdeal.mul_mem_left _ _ _ h


instance : SMulMemClass (TwoSidedIdeal R) Rᵐᵒᵖ R where
  smul_mem _ _ h := TwoSidedIdeal.mul_mem_right _ _ _ h


instance : Add I where add x y := ⟨x.1 + y.1, I.add_mem x.2 y.2⟩


instance : Zero I where zero := ⟨0, I.zero_mem⟩


instance : SMul ℕ I where smul n x := ⟨n • x.1, I.nsmul_mem n x.2⟩


instance : Neg I where neg x := ⟨-x.1, I.neg_mem x.2⟩


instance : Sub I where sub x y := ⟨x.1 - y.1, I.sub_mem x.2 y.2⟩


instance : SMul ℤ I where smul n x := ⟨n • x.1, I.zsmul_mem n x.2⟩


instance addCommGroup : AddCommGroup I :=
  Function.Injective.addCommGroup _ Subtype.coe_injective
    rfl (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


/-- The coercion into the ring as a `AddMonoidHom` -/
@[simp]
def coeAddMonoidHom : I →+ R where
  toFun := (↑)
  map_zero' := rfl
  map_add' _ _ := rfl


