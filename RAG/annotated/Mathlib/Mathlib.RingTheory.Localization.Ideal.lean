variable {M S} in
theorem mk'_mem_iff {x} {y : M} {I : Ideal S} : mk' S x y ∈ I ↔ algebraMap R S x ∈ I := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    I : Ideal S
    ⊢ Iff (Membership.mem I (IsLocalization.mk' S x y)) (Membership.mem I ((algebr …
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I (IsLocalization.mk' S x y)
      ⊢ Membership.mem I ((algebraMap R S) x)
    -/
  · rw [← mk'_spec S x y, mul_comm]
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I (IsLocalization.mk' S x y)
      ⊢ Membership.mem I (HMul.hMul ((algebraMap R S) ↑y) (IsLocalization.mk' S x y))
    -/
    exact I.mul_mem_left ((algebraMap R S) y) h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I ((algebraMap R S) x)
      ⊢ Membership.mem I (IsLocalization.mk' S x y)
    -/
  · rw [← mk'_spec S x y] at h
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I (HMul.hMul (IsLocalization.mk' S x y) ((algebraMap R S) ↑ …
      ⊢ Membership.mem I (IsLocalization.mk' S x y)
    -/
    obtain ⟨b, hb⟩ := isUnit_iff_exists_inv.1 (map_units S y)
    /-
      case mpr.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I (HMul.hMul (IsLocalization.mk' S x y) ((algebraMap R S) ↑ …
      b : S
      hb : Eq (HMul.hMul ((algebraMap R S) ↑y) b) 1
      ⊢ Membership.mem I (IsLocalization.mk' S x y)
    -/
    have := I.mul_mem_left b h
    /-
      case mpr.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      y : Subtype fun x => Membership.mem M x
      I : Ideal S
      h : Membership.mem I (HMul.hMul (IsLocalization.mk' S x y) ((algebraMap R S) ↑ …
      b : S
      hb : Eq (HMul.hMul ((algebraMap R S) ↑y) b) 1
      this : Membership.mem I (HMul.hMul b (HMul.hMul (IsLocalization.mk' S x y) ((a …
      ⊢ Membership.mem I (IsLocalization.mk' S x y)
    -/
    rwa [mul_comm, mul_assoc, hb, mul_one] at this
    /-
      🎉 no goals
    -/


/-- Explicit characterization of the ideal given by `Ideal.map (algebraMap R S) I`.
In practice, this ideal differs only in that the carrier set is defined explicitly.
This definition is only meant to be used in proving `mem_map_algebraMap_iff`,
and any proof that needs to refer to the explicit carrier set should use that theorem. -/
-- TODO: golf this using `Submodule.localized'`
private def map_ideal (I : Ideal R) : Ideal S where
  carrier := { z : S | ∃ x : I × M, z * algebraMap R S x.2 = algebraMap R S x.1 }
                           /-
                             R : Type u_1
                             inst✝³ : CommSemiring R
                             M : Submonoid R
                             S : Type u_2
                             inst✝² : CommSemiring S
                             inst✝¹ : Algebra R S
                             inst✝ : IsLocalization M S
                             I : Ideal R
                             ⊢ Eq (HMul.hMul 0 ((algebraMap R S) ↑{ fst := 0, snd := 1 }.2)) ((algebraMap R …
                           -/
  zero_mem' := ⟨⟨0, 1⟩, by simp⟩
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      ⊢ ∀ {a b : S}, Membership.mem (setOf fun z => Exists fun x => Eq (HMul.hMul z  …
    -/
                           /-
                             🎉 no goals
                           -/
  add_mem' := by
    rintro a b ⟨a', ha⟩ ⟨b', hb⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      a b : S
      a' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      ha : Eq (HMul.hMul a ((algebraMap R S) ↑a'.2)) ((algebraMap R S) ↑a'.1)
      b' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hb : Eq (HMul.hMul b ((algebraMap R S) ↑b'.2)) ((algebraMap R S) ↑b'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HAdd.hAdd (HMul.hMul ↑a'.2 ↑b'.1)  …
      ⊢ Membership.mem (setOf fun z => Exists fun x => Eq (HMul.hMul z ((algebraMap  …
    -/
    let Z : { x // x ∈ I } := ⟨(a'.2 : R) * (b'.1 : R) + (b'.2 : R) * (a'.1 : R),
    /-
      case h
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      a b : S
      a' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      ha : Eq (HMul.hMul a ((algebraMap R S) ↑a'.2)) ((algebraMap R S) ↑a'.1)
      b' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hb : Eq (HMul.hMul b ((algebraMap R S) ↑b'.2)) ((algebraMap R S) ↑b'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HAdd.hAdd (HMul.hMul ↑a'.2 ↑b'.1)  …
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) ((algebraMap R S) ↑{ fst := Z, snd := HMul.hMu …
    -/
      I.add_mem (I.mul_mem_left _ b'.1.2) (I.mul_mem_left _ a'.1.2)⟩
    use ⟨Z, a'.2 * b'.2⟩
    simp only [Z, RingHom.map_add, Submodule.coe_mk, Submonoid.coe_mul, RingHom.map_mul]
    /-
      case h
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      a b : S
      a' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      ha : Eq (HMul.hMul a ((algebraMap R S) ↑a'.2)) ((algebraMap R S) ↑a'.1)
      b' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hb : Eq (HMul.hMul b ((algebraMap R S) ↑b'.2)) ((algebraMap R S) ↑b'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HAdd.hAdd (HMul.hMul ↑a'.2 ↑b'.1)  …
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap R S) ↑a'.1) ((algebraMap R S) ↑b'.2))  …
    -/
    rw [add_mul, ← mul_assoc a, ha, mul_comm (algebraMap R S a'.2) (algebraMap R S b'.2), ←
    /-
      🎉 no goals
    -/
      mul_assoc b, hb]
    ring
  smul_mem' := by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      ⊢ ∀ (c : S) {x : S}, Membership.mem { carrier := setOf fun z => Exists fun x = …
    -/
    rintro c x ⟨x', hx⟩
    /-
      case intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      c x : S
      x' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hx : Eq (HMul.hMul x ((algebraMap R S) ↑x'.2)) ((algebraMap R S) ↑x'.1)
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => Eq (HMul.hMul z ( …
    -/
    obtain ⟨c', hc⟩ := IsLocalization.surj M c
    /-
      case intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      c x : S
      x' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hx : Eq (HMul.hMul x ((algebraMap R S) ↑x'.2)) ((algebraMap R S) ↑x'.1)
      c' : Prod R (Subtype fun x => Membership.mem M x)
      hc : Eq (HMul.hMul c ((algebraMap R S) ↑c'.2)) ((algebraMap R S) c'.1)
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => Eq (HMul.hMul z ( …
    -/
    let Z : { x // x ∈ I } := ⟨c'.1 * x'.1, I.mul_mem_left c'.1 x'.1.2⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      c x : S
      x' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hx : Eq (HMul.hMul x ((algebraMap R S) ↑x'.2)) ((algebraMap R S) ↑x'.1)
      c' : Prod R (Subtype fun x => Membership.mem M x)
      hc : Eq (HMul.hMul c ((algebraMap R S) ↑c'.2)) ((algebraMap R S) c'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HMul.hMul c'.1 ↑x'.1, ⋯⟩
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => Eq (HMul.hMul z ( …
    -/
    use ⟨Z, c'.2 * x'.2⟩
    /-
      case h
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      c x : S
      x' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hx : Eq (HMul.hMul x ((algebraMap R S) ↑x'.2)) ((algebraMap R S) ↑x'.1)
      c' : Prod R (Subtype fun x => Membership.mem M x)
      hc : Eq (HMul.hMul c ((algebraMap R S) ↑c'.2)) ((algebraMap R S) c'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HMul.hMul c'.1 ↑x'.1, ⋯⟩
      ⊢ Eq (HMul.hMul (HSMul.hSMul c x) ((algebraMap R S) ↑{ fst := Z, snd := HMul.h …
    -/
    simp only [Z, ← hx, ← hc, smul_eq_mul, Submodule.coe_mk, Submonoid.coe_mul, RingHom.map_mul]
    /-
      case h
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      c x : S
      x' : Prod (Subtype fun x => Membership.mem I x) (Subtype fun x => Membership.m …
      hx : Eq (HMul.hMul x ((algebraMap R S) ↑x'.2)) ((algebraMap R S) ↑x'.1)
      c' : Prod R (Subtype fun x => Membership.mem M x)
      hc : Eq (HMul.hMul c ((algebraMap R S) ↑c'.2)) ((algebraMap R S) c'.1)
      Z : Subtype fun x => Membership.mem I x := ⟨HMul.hMul c'.1 ↑x'.1, ⋯⟩
      ⊢ Eq (HMul.hMul (HMul.hMul c x) (HMul.hMul ((algebraMap R S) ↑c'.2) ((algebraM …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem mem_map_algebraMap_iff {I : Ideal R} {z} : z ∈ Ideal.map (algebraMap R S) I ↔
    ∃ x : I × M, z * algebraMap R S x.2 = algebraMap R S x.1 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    z : S
    ⊢ Iff (Membership.mem (Ideal.map (algebraMap R S) I) z) (Exists fun x => Eq (H …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z : S
      ⊢ Membership.mem (Ideal.map (algebraMap R S) I) z → Exists fun x => Eq (HMul.h …
    -/
  · change _ → z ∈ map_ideal M S I
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z : S
      ⊢ Membership.mem (Ideal.map (algebraMap R S) I) z → Membership.mem (IsLocaliza …
    -/
    refine fun h => Ideal.mem_sInf.1 h fun z hz => ?_
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z✝ : S
      h : Membership.mem (Ideal.map (algebraMap R S) I) z✝
      z : S
      hz : Membership.mem (Set.image ⇑(algebraMap R S) ↑I) z
      ⊢ Membership.mem (↑(IsLocalization.map_ideal M S I)) z
    -/
    obtain ⟨y, hy⟩ := hz
    /-
      case mp.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z✝ : S
      h : Membership.mem (Ideal.map (algebraMap R S) I) z✝
      z : S
      y : R
      hy : And (Membership.mem (↑I) y) (Eq ((algebraMap R S) y) z)
      ⊢ Membership.mem (↑(IsLocalization.map_ideal M S I)) z
    -/
    let Z : { x // x ∈ I } := ⟨y, hy.left⟩
    /-
      case mp.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z✝ : S
      h : Membership.mem (Ideal.map (algebraMap R S) I) z✝
      z : S
      y : R
      hy : And (Membership.mem (↑I) y) (Eq ((algebraMap R S) y) z)
      Z : Subtype fun x => Membership.mem I x := ⟨y, ⋯⟩
      ⊢ Membership.mem (↑(IsLocalization.map_ideal M S I)) z
    -/
    use ⟨Z, 1⟩
    /-
      case h
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z✝ : S
      h : Membership.mem (Ideal.map (algebraMap R S) I) z✝
      z : S
      y : R
      hy : And (Membership.mem (↑I) y) (Eq ((algebraMap R S) y) z)
      Z : Subtype fun x => Membership.mem I x := ⟨y, ⋯⟩
      ⊢ Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := Z, snd := 1 }.2)) ((algebraMap R …
    -/
    simp [Z, hy.right]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z : S
      ⊢ (Exists fun x => Eq (HMul.hMul z ((algebraMap R S) ↑x.2)) ((algebraMap R S)  …
    -/
  · rintro ⟨⟨a, s⟩, h⟩
    /-
      case mpr.intro.mk
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z : S
      a : Subtype fun x => Membership.mem I x
      s : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := a, snd := s }.2)) ((algebraMap …
      ⊢ Membership.mem (Ideal.map (algebraMap R S) I) z
    -/
    rw [← Ideal.unit_mul_mem_iff_mem _ (map_units S s), mul_comm]
    /-
      case mpr.intro.mk
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      z : S
      a : Subtype fun x => Membership.mem I x
      s : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := a, snd := s }.2)) ((algebraMap …
      ⊢ Membership.mem (Ideal.map (algebraMap R S) I) (HMul.hMul z ((algebraMap R S) …
    -/
    exact h.symm ▸ Ideal.mem_map_of_mem _ a.2
    /-
      🎉 no goals
    -/


lemma mk'_mem_map_algebraMap_iff (I : Ideal R) (x : R) (s : M) :
    IsLocalization.mk' S x s ∈ I.map (algebraMap R S) ↔ ∃ s ∈ M, s * x ∈ I := by
  rw [← Ideal.unit_mul_mem_iff_mem _ (IsLocalization.map_units S s), IsLocalization.mk'_spec',
    IsLocalization.mem_map_algebraMap_iff M]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    x : R
    s : Subtype fun x => Membership.mem M x
    ⊢ Iff (Exists fun x_1 => Eq (HMul.hMul ((algebraMap R S) x) ((algebraMap R S)  …
  -/
  simp_rw [← map_mul, IsLocalization.eq_iff_exists M, mul_comm x, ← mul_assoc, ← Submonoid.coe_mul]
  exact ⟨fun ⟨⟨y, t⟩, c, h⟩ ↦ ⟨_, (c * t).2, h ▸ I.mul_mem_left c.1 y.2⟩, fun ⟨s, hs, h⟩ ↦
    ⟨⟨⟨_, h⟩, ⟨s, hs⟩⟩, 1, by simp⟩⟩


include M in
theorem map_comap (J : Ideal S) :
    Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S) J) = J :=
  le_antisymm (Ideal.map_le_iff_le_comap.2 le_rfl) fun x hJ => by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      J : Ideal S
      x : S
      hJ : Membership.mem J x
      ⊢ Membership.mem (Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S) J)) x
    -/
    obtain ⟨r, s, hx⟩ := mk'_surjective M x
    /-
      case intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      J : Ideal S
      x : S
      hJ : Membership.mem J x
      r : R
      s : Subtype fun x => Membership.mem M x
      hx : Eq (IsLocalization.mk' S r s) x
      ⊢ Membership.mem (Ideal.map (algebraMap R S) (Ideal.comap (algebraMap R S) J)) x
    -/
    rw [← hx] at hJ ⊢
    exact
      Ideal.mul_mem_right _ _
        (Ideal.mem_map_of_mem _
          (show (algebraMap R S) r ∈ J from
            mk'_spec S r s ▸ J.mul_mem_right ((algebraMap R S) s) hJ))


theorem comap_map_of_isPrime_disjoint (I : Ideal R) (hI : I.IsPrime) (hM : Disjoint (M : Set R) I) :
    Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S) I) = I := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hI : I.IsPrime
    hM : Disjoint ↑M ↑I
    ⊢ Eq (Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S) I)) I
  -/
  refine le_antisymm ?_ Ideal.le_comap_map
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hI : I.IsPrime
    hM : Disjoint ↑M ↑I
    ⊢ LE.le (Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S) I)) I
  -/
  refine (fun a ha => ?_)
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hI : I.IsPrime
    hM : Disjoint ↑M ↑I
    a : R
    ha : Membership.mem (Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S)  …
    ⊢ Membership.mem I a
  -/
  obtain ⟨⟨b, s⟩, h⟩ := (mem_map_algebraMap_iff M S).1 (Ideal.mem_comap.1 ha)
  replace h : algebraMap R S (s * a) = algebraMap R S b := by
    simpa only [← map_mul, mul_comm] using h
  /-
    case intro.mk
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hI : I.IsPrime
    hM : Disjoint ↑M ↑I
    a : R
    ha : Membership.mem (Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S)  …
    b : Subtype fun x => Membership.mem I x
    s : Subtype fun x => Membership.mem M x
    h : Eq ((algebraMap R S) (HMul.hMul (↑s) a)) ((algebraMap R S) ↑b)
    ⊢ Membership.mem I a
  -/
  obtain ⟨c, hc⟩ := (eq_iff_exists M S).1 h
  have : ↑c * ↑s * a ∈ I := by
    rw [mul_assoc, hc]
    exact I.mul_mem_left c b.2
  /-
    case intro.mk.intro
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hI : I.IsPrime
    hM : Disjoint ↑M ↑I
    a : R
    ha : Membership.mem (Ideal.comap (algebraMap R S) (Ideal.map (algebraMap R S)  …
    b : Subtype fun x => Membership.mem I x
    s : Subtype fun x => Membership.mem M x
    h : Eq ((algebraMap R S) (HMul.hMul (↑s) a)) ((algebraMap R S) ↑b)
    c : Subtype fun x => Membership.mem M x
    hc : Eq (HMul.hMul (↑c) (HMul.hMul (↑s) a)) (HMul.hMul ↑c ↑b)
    this : Membership.mem I (HMul.hMul (HMul.hMul ↑c ↑s) a)
    ⊢ Membership.mem I a
  -/
  exact (hI.mem_or_mem this).resolve_left fun hsc => hM.le_bot ⟨(c * s).2, hsc⟩
  /-
    🎉 no goals
  -/


/-- If `S` is the localization of `R` at a submonoid, the ordering of ideals of `S` is
embedded in the ordering of ideals of `R`. -/
def orderEmbedding : Ideal S ↪o Ideal R where
  toFun J := Ideal.comap (algebraMap R S) J
  inj' := Function.LeftInverse.injective (map_comap M S)
  map_rel_iff' := by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      ⊢ ∀ {a b : Ideal S}, Iff (LE.le ({ toFun := fun J => Ideal.comap (algebraMap R …
    -/
    rintro J₁ J₂
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      J₁ J₂ : Ideal S
      ⊢ Iff (LE.le ({ toFun := fun J => Ideal.comap (algebraMap R S) J, inj' := ⋯ }  …
    -/
    constructor
      /-
        case mp
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J₁ J₂ : Ideal S
        ⊢ LE.le ({ toFun := fun J => Ideal.comap (algebraMap R S) J, inj' := ⋯ } J₁) ( …
      -/
    · exact fun hJ => (map_comap M S) J₁ ▸ (map_comap M S) J₂ ▸ Ideal.map_mono hJ
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J₁ J₂ : Ideal S
        ⊢ LE.le J₁ J₂ → LE.le ({ toFun := fun J => Ideal.comap (algebraMap R S) J, inj …
      -/
    · exact fun hJ => Ideal.comap_mono hJ
      /-
        🎉 no goals
      -/


/-- If `R` is a ring, then prime ideals in the localization at `M`
correspond to prime ideals in the original ring `R` that are disjoint from `M`.
This lemma gives the particular case for an ideal and its comap,
see `le_rel_iso_of_prime` for the more general relation isomorphism -/
theorem isPrime_iff_isPrime_disjoint (J : Ideal S) :
    J.IsPrime ↔
      (Ideal.comap (algebraMap R S) J).IsPrime ∧
        Disjoint (M : Set R) ↑(Ideal.comap (algebraMap R S) J) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    J : Ideal S
    ⊢ Iff J.IsPrime (And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(I …
  -/
  constructor
  · refine fun h =>
      ⟨⟨?_, ?_⟩,
        Set.disjoint_left.mpr fun m hm1 hm2 =>
          h.ne_top (Ideal.eq_top_of_isUnit_mem _ hm2 (map_units S ⟨m, hm1⟩))⟩
      /-
        case mp.refine_1
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        ⊢ Ne (Ideal.comap (algebraMap R S) J) Top.top
      -/
    · refine fun hJ => h.ne_top ?_
      /-
        case mp.refine_1
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        hJ : Eq (Ideal.comap (algebraMap R S) J) Top.top
        ⊢ Eq J Top.top
      -/
      rw [eq_top_iff, ← (orderEmbedding M S).le_iff_le]
      /-
        case mp.refine_1
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        hJ : Eq (Ideal.comap (algebraMap R S) J) Top.top
        ⊢ LE.le ((IsLocalization.orderEmbedding M S) Top.top) ((IsLocalization.orderEm …
      -/
      exact le_of_eq hJ.symm
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        ⊢ ∀ {x y : R}, Membership.mem (Ideal.comap (algebraMap R S) J) (HMul.hMul x y) …
      -/
    · intro x y hxy
      /-
        case mp.refine_2
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        x y : R
        hxy : Membership.mem (Ideal.comap (algebraMap R S) J) (HMul.hMul x y)
        ⊢ Or (Membership.mem (Ideal.comap (algebraMap R S) J) x) (Membership.mem (Idea …
      -/
      rw [Ideal.mem_comap, RingHom.map_mul] at hxy
      /-
        case mp.refine_2
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : J.IsPrime
        x y : R
        hxy : Membership.mem J (HMul.hMul ((algebraMap R S) x) ((algebraMap R S) y))
        ⊢ Or (Membership.mem (Ideal.comap (algebraMap R S) J) x) (Membership.mem (Idea …
      -/
      exact h.mem_or_mem hxy
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      J : Ideal S
      ⊢ And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (alg …
    -/
  · refine fun h => ⟨fun hJ => h.left.ne_top (eq_top_iff.2 ?_), ?_⟩
      /-
        case mpr.refine_1
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        hJ : Eq J Top.top
        ⊢ LE.le Top.top (Ideal.comap (algebraMap R S) J)
      -/
    · rwa [eq_top_iff, ← (orderEmbedding M S).le_iff_le] at hJ
      /-
        🎉 no goals
      -/
      /-
        case mpr.refine_2
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        ⊢ ∀ {x y : S}, Membership.mem J (HMul.hMul x y) → Or (Membership.mem J x) (Mem …
      -/
    · intro x y hxy
      /-
        case mpr.refine_2
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      obtain ⟨a, s, ha⟩ := mk'_surjective M x
      /-
        case mpr.refine_2.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      obtain ⟨b, t, hb⟩ := mk'_surjective M y
      /-
        case mpr.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        b : R
        t : Subtype fun x => Membership.mem M x
        hb : Eq (IsLocalization.mk' S b t) y
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      have : mk' S (a * b) (s * t) ∈ J := by rwa [mk'_mul, ha, hb]
      /-
        case mpr.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        b : R
        t : Subtype fun x => Membership.mem M x
        hb : Eq (IsLocalization.mk' S b t) y
        this : Membership.mem J (IsLocalization.mk' S (HMul.hMul a b) (HMul.hMul s t))
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      rw [mk'_mem_iff, ← Ideal.mem_comap] at this
      /-
        case mpr.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        b : R
        t : Subtype fun x => Membership.mem M x
        hb : Eq (IsLocalization.mk' S b t) y
        this : Membership.mem (Ideal.comap (algebraMap R S) J) (HMul.hMul a b)
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      have this₂ := (h.1).mul_mem_iff_mem_or_mem.1 this
      /-
        case mpr.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        b : R
        t : Subtype fun x => Membership.mem M x
        hb : Eq (IsLocalization.mk' S b t) y
        this : Membership.mem (Ideal.comap (algebraMap R S) J) (HMul.hMul a b)
        this₂ : Or (Membership.mem (Ideal.comap (algebraMap R S) J) a) (Membership.mem …
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      rw [Ideal.mem_comap, Ideal.mem_comap] at this₂
      /-
        case mpr.refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        inst✝ : IsLocalization M S
        J : Ideal S
        h : And (Ideal.comap (algebraMap R S) J).IsPrime (Disjoint ↑M ↑(Ideal.comap (a …
        x y : S
        hxy : Membership.mem J (HMul.hMul x y)
        a : R
        s : Subtype fun x => Membership.mem M x
        ha : Eq (IsLocalization.mk' S a s) x
        b : R
        t : Subtype fun x => Membership.mem M x
        hb : Eq (IsLocalization.mk' S b t) y
        this : Membership.mem (Ideal.comap (algebraMap R S) J) (HMul.hMul a b)
        this₂ : Or (Membership.mem J ((algebraMap R S) a)) (Membership.mem J ((algebra …
        ⊢ Or (Membership.mem J x) (Membership.mem J y)
      -/
      rwa [← ha, ← hb, mk'_mem_iff, mk'_mem_iff]
      /-
        🎉 no goals
      -/


/-- If `R` is a ring, then prime ideals in the localization at `M`
correspond to prime ideals in the original ring `R` that are disjoint from `M`.
This lemma gives the particular case for an ideal and its map,
see `le_rel_iso_of_prime` for the more general relation isomorphism, and the reverse implication -/
theorem isPrime_of_isPrime_disjoint (I : Ideal R) (hp : I.IsPrime) (hd : Disjoint (M : Set R) ↑I) :
    (Ideal.map (algebraMap R S) I).IsPrime := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hp : I.IsPrime
    hd : Disjoint ↑M ↑I
    ⊢ (Ideal.map (algebraMap R S) I).IsPrime
  -/
  rw [isPrime_iff_isPrime_disjoint M S, comap_map_of_isPrime_disjoint M S I hp hd]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    hp : I.IsPrime
    hd : Disjoint ↑M ↑I
    ⊢ And I.IsPrime (Disjoint ↑M ↑I)
  -/
  exact ⟨hp, hd⟩
  /-
    🎉 no goals
  -/


/-- If `R` is a ring, then prime ideals in the localization at `M`
correspond to prime ideals in the original ring `R` that are disjoint from `M` -/
def orderIsoOfPrime :
    { p : Ideal S // p.IsPrime } ≃o { p : Ideal R // p.IsPrime ∧ Disjoint (M : Set R) ↑p } where
  toFun p := ⟨Ideal.comap (algebraMap R S) p.1, (isPrime_iff_isPrime_disjoint M S p.1).1 p.2⟩
  invFun p := ⟨Ideal.map (algebraMap R S) p.1, isPrime_of_isPrime_disjoint M S p.1 p.2.1 p.2.2⟩
  left_inv J := Subtype.eq (map_comap M S J)
  right_inv I := Subtype.eq (comap_map_of_isPrime_disjoint M S I.1 I.2.1 I.2.2)
  map_rel_iff' := by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      ⊢ ∀ {a b : Subtype fun p => p.IsPrime}, Iff (LE.le ({ toFun := fun p => ⟨Ideal …
    -/
    rintro I I'
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I I' : Subtype fun p => p.IsPrime
      ⊢ Iff (LE.le ({ toFun := fun p => ⟨Ideal.comap (algebraMap R S) ↑p, ⋯⟩, invFun …
    -/
    constructor
    · exact (fun h => show I.val ≤ I'.val from map_comap M S I.val ▸
        map_comap M S I'.val ▸ Ideal.map_mono h)
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I I' : Subtype fun p => p.IsPrime
      ⊢ LE.le I I' → LE.le ({ toFun := fun p => ⟨Ideal.comap (algebraMap R S) ↑p, ⋯⟩ …
    -/
    exact fun h x hx => h hx
    /-
      🎉 no goals
    -/


include M in
/-- `quotientMap` applied to maximal ideals of a localization is `surjective`.
  The quotient by a maximal ideal is a field, so inverses to elements already exist,
  and the localization necessarily maps the equivalence class of the inverse in the localization -/
theorem surjective_quotientMap_of_maximal_of_localization {I : Ideal S} [I.IsPrime] {J : Ideal R}
    {H : J ≤ I.comap (algebraMap R S)} (hI : (I.comap (algebraMap R S)).IsMaximal) :
    Function.Surjective (Ideal.quotientMap I (algebraMap R S) H) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    I : Ideal S
    inst✝ : I.IsPrime
    J : Ideal R
    H : LE.le J (Ideal.comap (algebraMap R S) I)
    hI : (Ideal.comap (algebraMap R S) I).IsMaximal
    ⊢ Function.Surjective ⇑(Ideal.quotientMap I (algebraMap R S) H)
  -/
  intro s
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    I : Ideal S
    inst✝ : I.IsPrime
    J : Ideal R
    H : LE.le J (Ideal.comap (algebraMap R S) I)
    hI : (Ideal.comap (algebraMap R S) I).IsMaximal
    s : HasQuotient.Quotient S I
    ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) s
  -/
  obtain ⟨s, rfl⟩ := Ideal.Quotient.mk_surjective s
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    I : Ideal S
    inst✝ : I.IsPrime
    J : Ideal R
    H : LE.le J (Ideal.comap (algebraMap R S) I)
    hI : (Ideal.comap (algebraMap R S) I).IsMaximal
    s : S
    ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
  -/
  obtain ⟨r, ⟨m, hm⟩, rfl⟩ := mk'_surjective M s
  /-
    case intro.intro.intro.mk
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    I : Ideal S
    inst✝ : I.IsPrime
    J : Ideal R
    H : LE.le J (Ideal.comap (algebraMap R S) I)
    hI : (Ideal.comap (algebraMap R S) I).IsMaximal
    r m : R
    hm : Membership.mem M m
    ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
  -/
  by_cases hM : (Ideal.Quotient.mk (I.comap (algebraMap R S))) m = 0
  · have : I = ⊤ := by
      rw [Ideal.eq_top_iff_one]
      rw [Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_comap] at hM
      convert I.mul_mem_right (mk' S (1 : R) ⟨m, hm⟩) hM
      rw [← mk'_eq_mul_mk'_one, mk'_self]
    /-
      case pos
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : (Ideal.comap (algebraMap R S) I).IsMaximal
      r m : R
      hm : Membership.mem M m
      hM : Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0
      this : Eq I Top.top
      ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
    -/
    exact ⟨0, eq_comm.1 (by simp [Ideal.Quotient.eq_zero_iff_mem, this])⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : (Ideal.comap (algebraMap R S) I).IsMaximal
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
    -/
  · rw [Ideal.Quotient.maximal_ideal_iff_isField_quotient] at hI
    /-
      case neg
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
    -/
    obtain ⟨n, hn⟩ := hI.3 hM
    /-
      case neg.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      n : HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I)
      hn : Eq (HMul.hMul ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) n) 1
      ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
    -/
    obtain ⟨rn, rfl⟩ := Ideal.Quotient.mk_surjective n
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      rn : R
      hn : Eq (HMul.hMul ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) (( …
      ⊢ Exists fun a => Eq ((Ideal.quotientMap I (algebraMap R S) H) a) ((Ideal.Quot …
    -/
    refine ⟨(Ideal.Quotient.mk J) (r * rn), ?_⟩
    -- The rest of the proof is essentially just algebraic manipulations to prove the equality
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      rn : R
      hn : Eq (HMul.hMul ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) (( …
      ⊢ Eq ((Ideal.quotientMap I (algebraMap R S) H) ((Ideal.Quotient.mk J) (HMul.hM …
    -/
    replace hn := congr_arg (Ideal.quotientMap I (algebraMap R S) le_rfl) hn
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      rn : R
      hn : Eq ((Ideal.quotientMap I (algebraMap R S) ⋯) (HMul.hMul ((Ideal.Quotient. …
      ⊢ Eq ((Ideal.quotientMap I (algebraMap R S) H) ((Ideal.Quotient.mk J) (HMul.hM …
    -/
    rw [RingHom.map_one, RingHom.map_mul] at hn
    rw [Ideal.quotientMap_mk, ← sub_eq_zero, ← RingHom.map_sub, Ideal.Quotient.eq_zero_iff_mem, ←
      Ideal.Quotient.eq_zero_iff_mem, RingHom.map_sub, sub_eq_zero, mk'_eq_mul_mk'_one]
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      rn : R
      hn : Eq (HMul.hMul ((Ideal.quotientMap I (algebraMap R S) ⋯) ((Ideal.Quotient. …
      ⊢ Eq ((Ideal.Quotient.mk I) ((algebraMap R S) (HMul.hMul r rn))) ((Ideal.Quoti …
    -/
    simp only [mul_eq_mul_left_iff, RingHom.map_mul]
    refine
      Or.inl
        (mul_left_cancel₀ (M₀ := S ⧸ I)
          (fun hn =>
            hM
              (Ideal.Quotient.eq_zero_iff_mem.2
                (Ideal.mem_comap.2 (Ideal.Quotient.eq_zero_iff_mem.1 hn))))
          (_root_.trans hn ?_))
    /-
      case neg.intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : IsLocalization M S
      I : Ideal S
      inst✝ : I.IsPrime
      J : Ideal R
      H : LE.le J (Ideal.comap (algebraMap R S) I)
      hI : IsField (HasQuotient.Quotient R (Ideal.comap (algebraMap R S) I))
      r m : R
      hm : Membership.mem M m
      hM : Not (Eq ((Ideal.Quotient.mk (Ideal.comap (algebraMap R S) I)) m) 0)
      rn : R
      hn : Eq (HMul.hMul ((Ideal.quotientMap I (algebraMap R S) ⋯) ((Ideal.Quotient. …
      ⊢ Eq 1 (HMul.hMul ((Ideal.Quotient.mk I) ((algebraMap R S) m)) ((Ideal.Quotien …
    -/
    rw [← map_mul, ← mk'_eq_mul_mk'_one, mk'_self, RingHom.map_one]
    /-
      🎉 no goals
    -/


theorem bot_lt_comap_prime [IsDomain R] (hM : M ≤ R⁰) (p : Ideal S) [hpp : p.IsPrime]
    (hp0 : p ≠ ⊥) : ⊥ < Ideal.comap (algebraMap R S) p := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : IsDomain R
    hM : LE.le M (nonZeroDivisors R)
    p : Ideal S
    hpp : p.IsPrime
    hp0 : Ne p Bot.bot
    ⊢ LT.lt Bot.bot (Ideal.comap (algebraMap R S) p)
  -/
  haveI : IsDomain S := isDomain_of_le_nonZeroDivisors _ hM
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : IsDomain R
    hM : LE.le M (nonZeroDivisors R)
    p : Ideal S
    hpp : p.IsPrime
    hp0 : Ne p Bot.bot
    this : IsDomain S
    ⊢ LT.lt Bot.bot (Ideal.comap (algebraMap R S) p)
  -/
  rw [← Ideal.comap_bot_of_injective (algebraMap R S) (IsLocalization.injective _ hM)]
  convert (orderIsoOfPrime M S).lt_iff_lt.mpr (show (⟨⊥, Ideal.bot_prime⟩ :
    { p : Ideal S // p.IsPrime }) < ⟨p, hpp⟩ from hp0.bot_lt)


theorem ideal_eq_iInf_comap_map_away {S : Finset R} (hS : Ideal.span (α := R) S = ⊤) (I : Ideal R) :
    I = ⨅ f ∈ S, (I.map (algebraMap R (Localization.Away f))).comap
    (algebraMap R (Localization.Away f)) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    I : Ideal R
    ⊢ Eq I (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Localization.Aw …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      ⊢ LE.le I (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Localization …
    -/
  · simp only [le_iInf₂_iff, ← Ideal.map_le_iff_le_comap, le_refl, implies_true]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      ⊢ LE.le (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Localization.A …
    -/
  · intro x hx
    /-
      case a
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x : R
      hx : Membership.mem (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Lo …
      ⊢ Membership.mem I x
    -/
    apply Submodule.mem_of_span_eq_top_of_smul_pow_mem _ _ hS
    /-
      case a.H
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x : R
      hx : Membership.mem (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Lo …
      ⊢ ∀ (r : ↑↑S), Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑r) n …
    -/
    rintro ⟨s, hs⟩
    /-
      case a.H.mk
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x : R
      hx : Membership.mem (iInf fun f => iInf fun h => Ideal.comap (algebraMap R (Lo …
      s : R
      hs : Membership.mem (↑S) s
      ⊢ Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) n) x)
    -/
    simp only [Ideal.mem_iInf, Ideal.mem_comap] at hx
    obtain ⟨⟨y, ⟨_, n, rfl⟩⟩, e⟩ :=
      (IsLocalization.mem_map_algebraMap_iff (.powers s) _).mp (hx s hs)
    /-
      case a.H.mk.intro.mk.mk.intro
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n : Nat
      e : Eq (HMul.hMul ((algebraMap R (Localization.Away s)) x) ((algebraMap R (Loc …
      ⊢ Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) n) x)
    -/
    dsimp only at e
    /-
      case a.H.mk.intro.mk.mk.intro
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n : Nat
      e : Eq (HMul.hMul ((algebraMap R (Localization.Away s)) x) ((algebraMap R (Loc …
      ⊢ Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) n) x)
    -/
    rw [← map_mul, IsLocalization.eq_iff_exists (.powers s)] at e
    /-
      case a.H.mk.intro.mk.mk.intro
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n : Nat
      e : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul x (HPow.hPow s n))) (HMul.hM …
      ⊢ Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) n) x)
    -/
    obtain ⟨⟨_, m, rfl⟩, e⟩ := e
    /-
      case a.H.mk.intro.mk.mk.intro.intro.mk.intro
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n m : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow s x) m, ⋯⟩) (HMul.hMul x (HPow.hPow s …
      ⊢ Exists fun n => Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) n) x)
    -/
    use m + n
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n m : Nat
      e : Eq (HMul.hMul (↑⟨(fun x => HPow.hPow s x) m, ⋯⟩) (HMul.hMul x (HPow.hPow s …
      ⊢ Membership.mem I (HSMul.hSMul (HPow.hPow (↑⟨s, hs⟩) (HAdd.hAdd m n)) x)
    -/
    dsimp at e ⊢
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n m : Nat
      e : Eq (HMul.hMul (HPow.hPow s m) (HMul.hMul x (HPow.hPow s n))) (HMul.hMul (H …
      ⊢ Membership.mem I (HMul.hMul (HPow.hPow s (HAdd.hAdd m n)) x)
    -/
    rw [pow_add, mul_assoc, ← mul_comm x, e]
    /-
      case h
      R : Type u_1
      inst✝ : CommRing R
      S : Finset R
      hS : Eq (Ideal.span ↑S) Top.top
      I : Ideal R
      x s : R
      hs : Membership.mem (↑S) s
      hx : ∀ (i : R), Membership.mem S i → Membership.mem (Ideal.map (algebraMap R ( …
      y : Subtype fun x => Membership.mem I x
      n m : Nat
      e : Eq (HMul.hMul (HPow.hPow s m) (HMul.hMul x (HPow.hPow s n))) (HMul.hMul (H …
      ⊢ Membership.mem I (HMul.hMul (HPow.hPow s m) ↑y)
    -/
    exact I.mul_mem_left _ y.2
    /-
      🎉 no goals
    -/


