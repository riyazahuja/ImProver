/-- The Jacobson radical of `I` is the infimum of all maximal (left) ideals containing `I`. -/
def jacobson (I : Ideal R) : Ideal R :=
  sInf { J : Ideal R | I ≤ J ∧ IsMaximal J }


theorem le_jacobson : I ≤ jacobson I := fun _ hx => mem_sInf.mpr fun _ hJ => hJ.left hx


@[simp]
theorem jacobson_idem : jacobson (jacobson I) = jacobson I :=
  le_antisymm (sInf_le_sInf fun _ hJ => ⟨sInf_le hJ, hJ.2⟩) le_jacobson


@[simp]
theorem jacobson_top : jacobson (⊤ : Ideal R) = ⊤ :=
  eq_top_iff.2 le_jacobson


@[simp]
theorem jacobson_eq_top_iff : jacobson I = ⊤ ↔ I = ⊤ :=
  ⟨fun H =>
    by_contradiction fun hi => let ⟨M, hm, him⟩ := exists_le_maximal I hi
      lt_top_iff_ne_top.1
        (lt_of_le_of_lt (show jacobson I ≤ M from sInf_le ⟨him, hm⟩) <|
          lt_top_iff_ne_top.2 hm.ne_top) H,
    fun H => eq_top_iff.2 <| le_sInf fun _ ⟨hij, _⟩ => H ▸ hij⟩


theorem jacobson_eq_bot : jacobson I = ⊥ → I = ⊥ := fun h => eq_bot_iff.mpr (h ▸ le_jacobson)


theorem jacobson_eq_self_of_isMaximal [H : IsMaximal I] : I.jacobson = I :=
  le_antisymm (sInf_le ⟨le_of_eq rfl, H⟩) le_jacobson


instance (priority := 100) jacobson.isMaximal [H : IsMaximal I] : IsMaximal (jacobson I) :=
  ⟨⟨fun htop => H.1.1 (jacobson_eq_top_iff.1 htop), fun _ hJ =>
    H.1.2 _ (lt_of_le_of_lt le_jacobson hJ)⟩⟩


theorem mem_jacobson_iff {x : R} : x ∈ jacobson I ↔ ∀ y, ∃ z, z * y * x + z - 1 ∈ I :=
  ⟨fun hx y =>
    by_cases
      (fun hxy : I ⊔ span {y * x + 1} = ⊤ =>
        let ⟨p, hpi, q, hq, hpq⟩ := Submodule.mem_sup.1 ((eq_top_iff_one _).1 hxy)
        let ⟨r, hr⟩ := mem_span_singleton'.1 hq
        ⟨r, by
          /-
            R : Type u
            inst✝ : Ring R
            I : Ideal R
            x : R
            hx : Membership.mem I.jacobson x
            y : R
            hxy : Eq (Max.max I (Ideal.span (Singleton.singleton (HAdd.hAdd (HMul.hMul y x …
            p : R
            hpi : Membership.mem I p
            q : R
            hq : Membership.mem (Ideal.span (Singleton.singleton (HAdd.hAdd (HMul.hMul y x …
            hpq : Eq (HAdd.hAdd p q) 1
            r : R
            hr : Eq (HMul.hMul r (HAdd.hAdd (HMul.hMul y x) 1)) q
            ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul r y) x) r) 1)
          -/
          rw [mul_assoc, ← mul_add_one, hr, ← hpq, ← neg_sub, add_sub_cancel_right]
          /-
            R : Type u
            inst✝ : Ring R
            I : Ideal R
            x : R
            hx : Membership.mem I.jacobson x
            y : R
            hxy : Eq (Max.max I (Ideal.span (Singleton.singleton (HAdd.hAdd (HMul.hMul y x …
            p : R
            hpi : Membership.mem I p
            q : R
            hq : Membership.mem (Ideal.span (Singleton.singleton (HAdd.hAdd (HMul.hMul y x …
            hpq : Eq (HAdd.hAdd p q) 1
            r : R
            hr : Eq (HMul.hMul r (HAdd.hAdd (HMul.hMul y x) 1)) q
            ⊢ Membership.mem I (Neg.neg p)
          -/
          exact I.neg_mem hpi⟩)
          /-
            🎉 no goals
          -/
      fun hxy : I ⊔ span {y * x + 1} ≠ ⊤ => let ⟨M, hm1, hm2⟩ := exists_le_maximal _ hxy
      suffices x ∉ M from (this <| mem_sInf.1 hx ⟨le_trans le_sup_left hm2, hm1⟩).elim
      fun hxm => hm1.1.1 <| (eq_top_iff_one _).2 <| add_sub_cancel_left (y * x) 1 ▸
        M.sub_mem (le_sup_right.trans hm2 <| subset_span rfl) (M.mul_mem_left _ hxm),
    fun hx => mem_sInf.2 fun M ⟨him, hm⟩ => by_contradiction fun hxm =>
      let ⟨y, i, hi, df⟩ := hm.exists_inv hxm
      let ⟨z, hz⟩ := hx (-y)
      hm.1.1 <| (eq_top_iff_one _).2 <| sub_sub_cancel (z * -y * x + z) 1 ▸
        M.sub_mem (by
          rw [mul_assoc, ← mul_add_one, neg_mul, ← sub_eq_iff_eq_add.mpr df.symm, neg_sub,
            sub_add_cancel]
          /-
            R : Type u
            inst✝ : Ring R
            I : Ideal R
            x : R
            hx : ∀ (y : R), Exists fun z => Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.h …
            M : Ideal R
            x✝ : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) M
            him : LE.le I M
            hm : M.IsMaximal
            hxm : Not (Membership.mem M x)
            y i : R
            hi : Membership.mem M i
            df : Eq (HAdd.hAdd (HMul.hMul y x) i) 1
            z : R
            hz : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z (Neg.neg y …
            ⊢ Membership.mem M (HMul.hMul z i)
          -/
          exact M.mul_mem_left _ hi) <| him hz⟩
          /-
            🎉 no goals
          -/


theorem exists_mul_add_sub_mem_of_mem_jacobson {I : Ideal R} (r : R) (h : r ∈ jacobson I) :
    ∃ s, s * (r + 1) - 1 ∈ I := by
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson r
    ⊢ Exists fun s => Membership.mem I (HSub.hSub (HMul.hMul s (HAdd.hAdd r 1)) 1)
  -/
  cases' mem_jacobson_iff.1 h 1 with s hs
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson r
    s : R
    hs : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul s 1) r) s) 1)
    ⊢ Exists fun s => Membership.mem I (HSub.hSub (HMul.hMul s (HAdd.hAdd r 1)) 1)
  -/
  use s
  /-
    case h
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson r
    s : R
    hs : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul s 1) r) s) 1)
    ⊢ Membership.mem I (HSub.hSub (HMul.hMul s (HAdd.hAdd r 1)) 1)
  -/
  rw [mul_add, mul_one]
  /-
    case h
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson r
    s : R
    hs : Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul s 1) r) s) 1)
    ⊢ Membership.mem I (HSub.hSub (HAdd.hAdd (HMul.hMul s r) s) 1)
  -/
  simpa using hs
  /-
    🎉 no goals
  -/


theorem exists_mul_sub_mem_of_sub_one_mem_jacobson {I : Ideal R} (r : R) (h : r - 1 ∈ jacobson I) :
    ∃ s, s * r - 1 ∈ I := by
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson (HSub.hSub r 1)
    ⊢ Exists fun s => Membership.mem I (HSub.hSub (HMul.hMul s r) 1)
  -/
  convert exists_mul_add_sub_mem_of_mem_jacobson _ h
  /-
    case h.e'_2.h.h.e'_5.h.e'_5.h.e'_6
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    r : R
    h : Membership.mem I.jacobson (HSub.hSub r 1)
    x✝ : R
    ⊢ Eq r (HAdd.hAdd (HSub.hSub r 1) 1)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An ideal equals its Jacobson radical iff it is the intersection of a set of maximal ideals.
Allowing the set to include ⊤ is equivalent, and is included only to simplify some proofs. -/
theorem eq_jacobson_iff_sInf_maximal :
    I.jacobson = I ↔ ∃ M : Set (Ideal R), (∀ J ∈ M, IsMaximal J ∨ J = ⊤) ∧ I = sInf M := by
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    ⊢ Iff (Eq I.jacobson I) (Exists fun M => And (∀ (J : Ideal R), Membership.mem  …
  -/
  use fun hI => ⟨{ J : Ideal R | I ≤ J ∧ J.IsMaximal }, ⟨fun _ hJ => Or.inl hJ.right, hI.symm⟩⟩
  /-
    case mpr
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    ⊢ (Exists fun M => And (∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal ( …
  -/
  rintro ⟨M, hM, hInf⟩
  /-
    case mpr.intro.intro
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    M : Set (Ideal R)
    hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
    hInf : Eq I (InfSet.sInf M)
    ⊢ Eq I.jacobson I
  -/
  refine le_antisymm (fun x hx => ?_) le_jacobson
  /-
    case mpr.intro.intro
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    M : Set (Ideal R)
    hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
    hInf : Eq I (InfSet.sInf M)
    x : R
    hx : Membership.mem I.jacobson x
    ⊢ Membership.mem I x
  -/
  rw [hInf, mem_sInf]
  /-
    case mpr.intro.intro
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    M : Set (Ideal R)
    hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
    hInf : Eq I (InfSet.sInf M)
    x : R
    hx : Membership.mem I.jacobson x
    ⊢ ∀ ⦃I : Ideal R⦄, Membership.mem M I → Membership.mem I x
  -/
  intro I hI
  /-
    case mpr.intro.intro
    R : Type u
    inst✝ : Ring R
    I✝ : Ideal R
    M : Set (Ideal R)
    hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
    hInf : Eq I✝ (InfSet.sInf M)
    x : R
    hx : Membership.mem I✝.jacobson x
    I : Ideal R
    hI : Membership.mem M I
    ⊢ Membership.mem I x
  -/
  cases' hM I hI with is_max is_top
    /-
      case mpr.intro.intro.inl
      R : Type u
      inst✝ : Ring R
      I✝ : Ideal R
      M : Set (Ideal R)
      hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
      hInf : Eq I✝ (InfSet.sInf M)
      x : R
      hx : Membership.mem I✝.jacobson x
      I : Ideal R
      hI : Membership.mem M I
      is_max : I.IsMaximal
      ⊢ Membership.mem I x
    -/
  · exact (mem_sInf.1 hx) ⟨le_sInf_iff.1 (le_of_eq hInf) I hI, is_max⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.intro.inr
      R : Type u
      inst✝ : Ring R
      I✝ : Ideal R
      M : Set (Ideal R)
      hM : ∀ (J : Ideal R), Membership.mem M J → Or J.IsMaximal (Eq J Top.top)
      hInf : Eq I✝ (InfSet.sInf M)
      x : R
      hx : Membership.mem I✝.jacobson x
      I : Ideal R
      hI : Membership.mem M I
      is_top : Eq I Top.top
      ⊢ Membership.mem I x
    -/
  · exact is_top.symm ▸ Submodule.mem_top
    /-
      🎉 no goals
    -/


theorem eq_jacobson_iff_sInf_maximal' :
    I.jacobson = I ↔ ∃ M : Set (Ideal R), (∀ J ∈ M, ∀ (K : Ideal R), J < K → K = ⊤) ∧ I = sInf M :=
  eq_jacobson_iff_sInf_maximal.trans
    ⟨fun h =>
      let ⟨M, hM⟩ := h
      ⟨M,
        ⟨fun J hJ K hK =>
          Or.recOn (hM.1 J hJ) (fun h => h.1.2 K hK) fun h => eq_top_iff.2 (le_of_lt (h ▸ hK)),
          hM.2⟩⟩,
      fun h =>
      let ⟨M, hM⟩ := h
      ⟨M,
        ⟨fun J hJ =>
          Or.recOn (Classical.em (J = ⊤)) (fun h => Or.inr h) fun h => Or.inl ⟨⟨h, hM.1 J hJ⟩⟩,
          hM.2⟩⟩⟩


/-- An ideal `I` equals its Jacobson radical if and only if every element outside `I`
also lies outside of a maximal ideal containing `I`. -/
theorem eq_jacobson_iff_not_mem :
    I.jacobson = I ↔ ∀ x ∉ I, ∃ M : Ideal R, (I ≤ M ∧ M.IsMaximal) ∧ x ∉ M := by
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    ⊢ Iff (Eq I.jacobson I) (∀ (x : R), Not (Membership.mem I x) → Exists fun M => …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      ⊢ Eq I.jacobson I → ∀ (x : R), Not (Membership.mem I x) → Exists fun M => And  …
    -/
  · intro h x hx
    /-
      case mp
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : Eq I.jacobson I
      x : R
      hx : Not (Membership.mem I x)
      ⊢ Exists fun M => And (And (LE.le I M) M.IsMaximal) (Not (Membership.mem M x))
    -/
    rw [← h, Ideal.jacobson, mem_sInf] at hx
    /-
      case mp
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : Eq I.jacobson I
      x : R
      hx : Not (∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J. …
      ⊢ Exists fun M => And (And (LE.le I M) M.IsMaximal) (Not (Membership.mem M x))
    -/
    push_neg at hx
    /-
      case mp
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : Eq I.jacobson I
      x : R
      hx : Exists fun ⦃I_1⦄ => And (Membership.mem (setOf fun J => And (LE.le I J) J …
      ⊢ Exists fun M => And (And (LE.le I M) M.IsMaximal) (Not (Membership.mem M x))
    -/
    exact hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      ⊢ (∀ (x : R), Not (Membership.mem I x) → Exists fun M => And (And (LE.le I M)  …
    -/
  · refine fun h => le_antisymm (fun x hx => ?_) le_jacobson
    /-
      case mpr
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : ∀ (x : R), Not (Membership.mem I x) → Exists fun M => And (And (LE.le I M) …
      x : R
      hx : Membership.mem I.jacobson x
      ⊢ Membership.mem I x
    -/
    contrapose hx
    /-
      case mpr
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : ∀ (x : R), Not (Membership.mem I x) → Exists fun M => And (And (LE.le I M) …
      x : R
      hx : Not (Membership.mem I x)
      ⊢ Not (Membership.mem I.jacobson x)
    -/
    rw [Ideal.jacobson, mem_sInf]
    /-
      case mpr
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : ∀ (x : R), Not (Membership.mem I x) → Exists fun M => And (And (LE.le I M) …
      x : R
      hx : Not (Membership.mem I x)
      ⊢ Not (∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsM …
    -/
    push_neg
    /-
      case mpr
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      h : ∀ (x : R), Not (Membership.mem I x) → Exists fun M => And (And (LE.le I M) …
      x : R
      hx : Not (Membership.mem I x)
      ⊢ Exists fun ⦃I_1⦄ => And (Membership.mem (setOf fun J => And (LE.le I J) J.Is …
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


theorem map_jacobson_of_surjective {f : R →+* S} (hf : Function.Surjective f) :
    RingHom.ker f ≤ I → map f I.jacobson = (map f I).jacobson := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Ring S
    I : Ideal R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ LE.le (RingHom.ker f) I → Eq (Ideal.map f I.jacobson) (Ideal.map f I).jacobson
  -/
  intro h
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Ring S
    I : Ideal R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : LE.le (RingHom.ker f) I
    ⊢ Eq (Ideal.map f I.jacobson) (Ideal.map f I).jacobson
  -/
  unfold Ideal.jacobson
  -- Porting note: dot notation for `RingHom.ker` does not work
  have : ∀ J ∈ { J : Ideal R | I ≤ J ∧ J.IsMaximal }, RingHom.ker f ≤ J :=
    fun J hJ => le_trans h hJ.left
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Ring S
    I : Ideal R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : LE.le (RingHom.ker f) I
    this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
    ⊢ Eq (Ideal.map f (InfSet.sInf (setOf fun J => And (LE.le I J) J.IsMaximal)))  …
  -/
  refine Trans.trans (map_sInf hf this) (le_antisymm ?_ ?_)
  · refine
      sInf_le_sInf fun J hJ =>
        ⟨comap f J, ⟨⟨le_comap_of_map_le hJ.1, ?_⟩, map_comap_of_surjective f hf J⟩⟩
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      I : Ideal R
      f : RingHom R S
      hf : Function.Surjective ⇑f
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
      J : Ideal S
      hJ : Membership.mem (setOf fun J => And (LE.le (Ideal.map f I) J) J.IsMaximal) J
      ⊢ (Ideal.comap f J).IsMaximal
    -/
    haveI : J.IsMaximal := hJ.right
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      I : Ideal R
      f : RingHom R S
      hf : Function.Surjective ⇑f
      h : LE.le (RingHom.ker f) I
      this✝ : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMa …
      J : Ideal S
      hJ : Membership.mem (setOf fun J => And (LE.le (Ideal.map f I) J) J.IsMaximal) J
      this : J.IsMaximal
      ⊢ (Ideal.comap f J).IsMaximal
    -/
    exact comap_isMaximal_of_surjective f hf
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      I : Ideal R
      f : RingHom R S
      hf : Function.Surjective ⇑f
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
      ⊢ LE.le (InfSet.sInf (setOf fun J => And (LE.le (Ideal.map f I) J) J.IsMaximal …
    -/
  · refine sInf_le_sInf_of_subset_insert_top fun j hj => hj.recOn fun J hJ => ?_
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      I : Ideal R
      f : RingHom R S
      hf : Function.Surjective ⇑f
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J …
      J : Ideal R
      hJ : And (Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) J) (Eq ( …
      ⊢ Membership.mem (Insert.insert Top.top (setOf fun J => And (LE.le (Ideal.map  …
    -/
    rw [← hJ.2]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      I : Ideal R
      f : RingHom R S
      hf : Function.Surjective ⇑f
      h : LE.le (RingHom.ker f) I
      this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
      j : Ideal S
      hj : Membership.mem (Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J …
      J : Ideal R
      hJ : And (Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) J) (Eq ( …
      ⊢ Membership.mem (Insert.insert Top.top (setOf fun J => And (LE.le (Ideal.map  …
    -/
    cases' map_eq_top_or_isMaximal_of_surjective f hf hJ.left.right with htop hmax
      /-
        case refine_2.inl
        R : Type u
        S : Type v
        inst✝¹ : Ring R
        inst✝ : Ring S
        I : Ideal R
        f : RingHom R S
        hf : Function.Surjective ⇑f
        h : LE.le (RingHom.ker f) I
        this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
        j : Ideal S
        hj : Membership.mem (Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J …
        J : Ideal R
        hJ : And (Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) J) (Eq ( …
        htop : Eq (Ideal.map f J) Top.top
        ⊢ Membership.mem (Insert.insert Top.top (setOf fun J => And (LE.le (Ideal.map  …
      -/
    · exact htop.symm ▸ Set.mem_insert ⊤ _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        R : Type u
        S : Type v
        inst✝¹ : Ring R
        inst✝ : Ring S
        I : Ideal R
        f : RingHom R S
        hf : Function.Surjective ⇑f
        h : LE.le (RingHom.ker f) I
        this : ∀ (J : Ideal R), Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
        j : Ideal S
        hj : Membership.mem (Set.image (Ideal.map f) (setOf fun J => And (LE.le I J) J …
        J : Ideal R
        hJ : And (Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) J) (Eq ( …
        hmax : (Ideal.map f J).IsMaximal
        ⊢ Membership.mem (Insert.insert Top.top (setOf fun J => And (LE.le (Ideal.map  …
      -/
    · exact Set.mem_insert_of_mem ⊤ ⟨map_mono hJ.1.1, hmax⟩
      /-
        🎉 no goals
      -/


theorem map_jacobson_of_bijective {f : R →+* S} (hf : Function.Bijective f) :
    map f I.jacobson = (map f I).jacobson :=
  map_jacobson_of_surjective hf.right
    (le_trans (le_of_eq (f.injective_iff_ker_eq_bot.1 hf.left)) bot_le)


theorem comap_jacobson {f : R →+* S} {K : Ideal S} :
    comap f K.jacobson = sInf (comap f '' { J : Ideal S | K ≤ J ∧ J.IsMaximal }) :=
  Trans.trans (comap_sInf' f _) sInf_eq_iInf.symm


theorem comap_jacobson_of_surjective {f : R →+* S} (hf : Function.Surjective f) {K : Ideal S} :
    comap f K.jacobson = (comap f K).jacobson := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    K : Ideal S
    ⊢ Eq (Ideal.comap f K.jacobson) (Ideal.comap f K).jacobson
  -/
  unfold Ideal.jacobson
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    K : Ideal S
    ⊢ Eq (Ideal.comap f (InfSet.sInf (setOf fun J => And (LE.le K J) J.IsMaximal)) …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K : Ideal S
      ⊢ LE.le (Ideal.comap f (InfSet.sInf (setOf fun J => And (LE.le K J) J.IsMaxima …
    -/
  · rw [← top_inf_eq (sInf _), ← sInf_insert, comap_sInf', sInf_eq_iInf]
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K : Ideal S
      ⊢ LE.le (iInf fun I => iInf fun h => I) (iInf fun a => iInf fun h => a)
    -/
    refine iInf_le_iInf_of_subset fun J hJ => ?_
    have : comap f (map f J) = J :=
      Trans.trans (comap_map_of_surjective f hf J)
        (le_antisymm (sup_le_iff.2 ⟨le_of_eq rfl, le_trans (comap_mono bot_le) hJ.left⟩)
          le_sup_left)
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K : Ideal S
      J : Ideal R
      hJ : Membership.mem (setOf fun J => And (LE.le (Ideal.comap f K) J) J.IsMaxima …
      this : Eq (Ideal.comap f (Ideal.map f J)) J
      ⊢ Membership.mem (Set.image (Ideal.comap f) (Insert.insert Top.top (setOf fun  …
    -/
    cases' map_eq_top_or_isMaximal_of_surjective _ hf hJ.right with htop hmax
      /-
        case refine_1.inl
        R : Type u
        S : Type v
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        hf : Function.Surjective ⇑f
        K : Ideal S
        J : Ideal R
        hJ : Membership.mem (setOf fun J => And (LE.le (Ideal.comap f K) J) J.IsMaxima …
        this : Eq (Ideal.comap f (Ideal.map f J)) J
        htop : Eq (Ideal.map f J) Top.top
        ⊢ Membership.mem (Set.image (Ideal.comap f) (Insert.insert Top.top (setOf fun  …
      -/
    · exact ⟨⊤, Set.mem_insert ⊤ _, htop ▸ this⟩
      /-
        🎉 no goals
      -/
    · exact ⟨map f J, Set.mem_insert_of_mem _ ⟨le_map_of_comap_le_of_surjective f hf hJ.1, hmax⟩,
        this⟩
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K : Ideal S
      ⊢ LE.le (InfSet.sInf (setOf fun J => And (LE.le (Ideal.comap f K) J) J.IsMaxim …
    -/
  · simp_rw [comap_sInf, le_iInf_iff]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K : Ideal S
      ⊢ ∀ (i : Ideal S), Membership.mem (setOf fun J => And (LE.le K J) J.IsMaximal) …
    -/
    intros J hJ
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K J : Ideal S
      hJ : Membership.mem (setOf fun J => And (LE.le K J) J.IsMaximal) J
      ⊢ LE.le (InfSet.sInf (setOf fun J => And (LE.le (Ideal.comap f K) J) J.IsMaxim …
    -/
    haveI : J.IsMaximal := hJ.right
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      hf : Function.Surjective ⇑f
      K J : Ideal S
      hJ : Membership.mem (setOf fun J => And (LE.le K J) J.IsMaximal) J
      this : J.IsMaximal
      ⊢ LE.le (InfSet.sInf (setOf fun J => And (LE.le (Ideal.comap f K) J) J.IsMaxim …
    -/
    exact sInf_le ⟨comap_mono hJ.left, comap_isMaximal_of_surjective _ hf⟩
    /-
      🎉 no goals
    -/


@[mono]
theorem jacobson_mono {I J : Ideal R} : I ≤ J → I.jacobson ≤ J.jacobson := by
  /-
    R : Type u
    inst✝ : Ring R
    I J : Ideal R
    ⊢ LE.le I J → LE.le I.jacobson J.jacobson
  -/
  intro h x hx
  /-
    R : Type u
    inst✝ : Ring R
    I J : Ideal R
    h : LE.le I J
    x : R
    hx : Membership.mem I.jacobson x
    ⊢ Membership.mem J.jacobson x
  -/
  erw [mem_sInf] at hx ⊢
  /-
    R : Type u
    inst✝ : Ring R
    I J : Ideal R
    h : LE.le I J
    x : R
    hx : ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsMax …
    ⊢ ∀ ⦃I : Ideal R⦄, Membership.mem (setOf fun J_1 => And (LE.le J J_1) J_1.IsMa …
  -/
  exact fun K ⟨hK, hK_max⟩ => hx ⟨Trans.trans h hK, hK_max⟩
  /-
    🎉 no goals
  -/


/-- The Jacobson radical of a two-sided ideal is two-sided.

It is preferable to use `TwoSidedIdeal.jacobson` instead of this lemma. -/
theorem jacobson_mul_mem_right {I : Ideal R}
    (mul_mem_right : ∀ {x y}, x ∈ I → x * y ∈ I) :
    ∀ {x y}, x ∈ I.jacobson → x * y ∈ I.jacobson := by
  -- Proof generalized from
  -- https://ysharifi.wordpress.com/2022/08/16/the-jacobson-radical-definition-and-basic-results/
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    ⊢ ∀ {x y : R}, Membership.mem I.jacobson x → Membership.mem I.jacobson (HMul.h …
  -/
  intro x r xJ
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    ⊢ Membership.mem I.jacobson (HMul.hMul x r)
  -/
  apply mem_sInf.mpr
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    ⊢ ∀ ⦃I_1 : Ideal R⦄, Membership.mem (setOf fun J => And (LE.le I J) J.IsMaxima …
  -/
  intro 𝔪 𝔪_mem
  /-
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    𝔪 : Ideal R
    𝔪_mem : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) 𝔪
    ⊢ Membership.mem 𝔪 (HMul.hMul x r)
  -/
  by_cases r𝔪 : r ∈ 𝔪
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      I : Ideal R
      mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
      x r : R
      xJ : Membership.mem I.jacobson x
      𝔪 : Ideal R
      𝔪_mem : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) 𝔪
      r𝔪 : Membership.mem 𝔪 r
      ⊢ Membership.mem 𝔪 (HMul.hMul x r)
    -/
  · apply 𝔪.smul_mem _ r𝔪
    /-
      🎉 no goals
    -/
  -- 𝔪₀ := { a : R | a*r ∈ 𝔪 }
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    𝔪 : Ideal R
    𝔪_mem : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) 𝔪
    r𝔪 : Not (Membership.mem 𝔪 r)
    ⊢ Membership.mem 𝔪 (HMul.hMul x r)
  -/
  let 𝔪₀ : Ideal R := Submodule.comap (DistribMulAction.toLinearMap R (S := Rᵐᵒᵖ) R (.op r)) 𝔪
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    𝔪 : Ideal R
    𝔪_mem : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) 𝔪
    r𝔪 : Not (Membership.mem 𝔪 r)
    𝔪₀ : Ideal R := Submodule.comap (DistribMulAction.toLinearMap R R (MulOpposite …
    ⊢ Membership.mem 𝔪 (HMul.hMul x r)
  -/
  suffices x ∈ 𝔪₀ by simpa [𝔪₀] using this
  have I𝔪₀ : I ≤ 𝔪₀ := fun i iI =>
    𝔪_mem.left (mul_mem_right iI)
  have 𝔪₀_maximal : IsMaximal 𝔪₀ := by
    refine isMaximal_iff.mpr ⟨
      fun h => r𝔪 (by simpa [𝔪₀] using h),
      fun J b 𝔪₀J b𝔪₀ bJ => ?_⟩
    let K : Ideal R := Ideal.span {b*r} ⊔ 𝔪
    have ⟨s, y, y𝔪, sbyr⟩ :=
      mem_span_singleton_sup.mp <|
        mul_mem_left _ r <|
          (isMaximal_iff.mp 𝔪_mem.right).right K (b*r)
          le_sup_right b𝔪₀
          (mem_sup_left <| mem_span_singleton_self _)
    have : 1 - s*b ∈ 𝔪₀ := by
      rw [mul_one, add_comm, ← eq_sub_iff_add_eq] at sbyr
      rw [sbyr, ← mul_assoc] at y𝔪
      simp [𝔪₀, sub_mul, y𝔪]
    have : 1 - s*b + s*b ∈ J := by
      apply add_mem (𝔪₀J this) (J.mul_mem_left _ bJ)
    simpa using this
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    I : Ideal R
    mul_mem_right : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul  …
    x r : R
    xJ : Membership.mem I.jacobson x
    𝔪 : Ideal R
    𝔪_mem : Membership.mem (setOf fun J => And (LE.le I J) J.IsMaximal) 𝔪
    r𝔪 : Not (Membership.mem 𝔪 r)
    𝔪₀ : Ideal R := Submodule.comap (DistribMulAction.toLinearMap R R (MulOpposite …
    I𝔪₀ : LE.le I 𝔪₀
    𝔪₀_maximal : 𝔪₀.IsMaximal
    ⊢ Membership.mem 𝔪₀ x
  -/
  exact mem_sInf.mp xJ ⟨I𝔪₀, 𝔪₀_maximal⟩
  /-
    🎉 no goals
  -/


theorem radical_le_jacobson : radical I ≤ jacobson I :=
  le_sInf fun _ hJ => (radical_eq_sInf I).symm ▸ sInf_le ⟨hJ.left, IsMaximal.isPrime hJ.right⟩


theorem isRadical_of_eq_jacobson (h : jacobson I = I) : I.IsRadical :=
  radical_le_jacobson.trans h.le


lemma isRadical_jacobson (I : Ideal R) : I.jacobson.IsRadical :=
  isRadical_of_eq_jacobson jacobson_idem


theorem isUnit_of_sub_one_mem_jacobson_bot (r : R) (h : r - 1 ∈ jacobson (⊥ : Ideal R)) :
    IsUnit r := by
  /-
    R : Type u
    inst✝ : CommRing R
    r : R
    h : Membership.mem Bot.bot.jacobson (HSub.hSub r 1)
    ⊢ IsUnit r
  -/
  cases' exists_mul_sub_mem_of_sub_one_mem_jacobson r h with s hs
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    r : R
    h : Membership.mem Bot.bot.jacobson (HSub.hSub r 1)
    s : R
    hs : Membership.mem Bot.bot (HSub.hSub (HMul.hMul s r) 1)
    ⊢ IsUnit r
  -/
  rw [mem_bot, sub_eq_zero, mul_comm] at hs
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    r : R
    h : Membership.mem Bot.bot.jacobson (HSub.hSub r 1)
    s : R
    hs : Eq (HMul.hMul r s) 1
    ⊢ IsUnit r
  -/
  exact isUnit_of_mul_eq_one _ _ hs
  /-
    🎉 no goals
  -/


theorem mem_jacobson_bot {x : R} : x ∈ jacobson (⊥ : Ideal R) ↔ ∀ y, IsUnit (x * y + 1) :=
  ⟨fun hx y =>
    let ⟨z, hz⟩ := (mem_jacobson_iff.1 hx) y
    isUnit_iff_exists_inv.2
             /-
               R : Type u
               inst✝ : CommRing R
               x : R
               hx : Membership.mem Bot.bot.jacobson x
               y z : R
               hz : Membership.mem Bot.bot (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul z y) x …
               ⊢ Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x y) 1) z) 1
             -/
      ⟨z, by rwa [add_mul, one_mul, ← sub_eq_zero, mul_right_comm, mul_comm _ z, mul_right_comm]⟩,
             /-
               🎉 no goals
             -/
    fun h =>
    mem_jacobson_iff.mpr fun y =>
      let ⟨b, hb⟩ := isUnit_iff_exists_inv.1 (h y)
                                           /-
                                             R : Type u
                                             inst✝ : CommRing R
                                             x : R
                                             h : ∀ (y : R), IsUnit (HAdd.hAdd (HMul.hMul x y) 1)
                                             y b : R
                                             hb : Eq (HMul.hMul (HAdd.hAdd (HMul.hMul x y) 1) b) 1
                                             ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HMul.hMul b y) x) b) (HMul.hMul (HAdd.h …
                                           -/
      ⟨b, (Submodule.mem_bot R).2 (hb ▸ by ring)⟩⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- An ideal `I` of `R` is equal to its Jacobson radical if and only if
the Jacobson radical of the quotient ring `R/I` is the zero ideal -/
-- Porting note: changed `Quotient.mk'` to ``
theorem jacobson_eq_iff_jacobson_quotient_eq_bot :
    I.jacobson = I ↔ jacobson (⊥ : Ideal (R ⧸ I)) = ⊥ := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff (Eq I.jacobson I) (Eq Bot.bot.jacobson Bot.bot)
  -/
  have hf : Function.Surjective (Ideal.Quotient.mk I) := Submodule.Quotient.mk_surjective I
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
    ⊢ Iff (Eq I.jacobson I) (Eq Bot.bot.jacobson Bot.bot)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      ⊢ Eq I.jacobson I → Eq Bot.bot.jacobson Bot.bot
    -/
  · intro h
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq I.jacobson I
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    replace h := congr_arg (Ideal.map (Ideal.Quotient.mk I)) h
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq (Ideal.map (Ideal.Quotient.mk I) I.jacobson) (Ideal.map (Ideal.Quotient …
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    rw [map_jacobson_of_surjective hf (le_of_eq mk_ker)] at h
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq (Ideal.map (Ideal.Quotient.mk I) I).jacobson (Ideal.map (Ideal.Quotient …
      ⊢ Eq Bot.bot.jacobson Bot.bot
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      ⊢ Eq Bot.bot.jacobson Bot.bot → Eq I.jacobson I
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq Bot.bot.jacobson Bot.bot
      ⊢ Eq I.jacobson I
    -/
    replace h := congr_arg (comap (Ideal.Quotient.mk I)) h
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq (Ideal.comap (Ideal.Quotient.mk I) Bot.bot.jacobson) (Ideal.comap (Idea …
      ⊢ Eq I.jacobson I
    -/
    rw [comap_jacobson_of_surjective hf, ← RingHom.ker_eq_comap_bot (Ideal.Quotient.mk I)] at h
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq (RingHom.ker (Ideal.Quotient.mk I)).jacobson (RingHom.ker (Ideal.Quotie …
      ⊢ Eq I.jacobson I
    -/
    simpa using h
    /-
      🎉 no goals
    -/


/-- The standard radical and Jacobson radical of an ideal `I` of `R` are equal if and only if
the nilradical and Jacobson radical of the quotient ring `R/I` coincide -/
-- Porting note: changed `Quotient.mk'` to ``
theorem radical_eq_jacobson_iff_radical_quotient_eq_jacobson_bot :
    I.radical = I.jacobson ↔ radical (⊥ : Ideal (R ⧸ I)) = jacobson ⊥ := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff (Eq I.radical I.jacobson) (Eq Bot.bot.radical Bot.bot.jacobson)
  -/
  have hf : Function.Surjective (Ideal.Quotient.mk I) := Submodule.Quotient.mk_surjective I
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
    ⊢ Iff (Eq I.radical I.jacobson) (Eq Bot.bot.radical Bot.bot.jacobson)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      ⊢ Eq I.radical I.jacobson → Eq Bot.bot.radical Bot.bot.jacobson
    -/
  · intro h
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq I.radical I.jacobson
      ⊢ Eq Bot.bot.radical Bot.bot.jacobson
    -/
    have := congr_arg (map (Ideal.Quotient.mk I)) h
    rw [map_radical_of_surjective hf (le_of_eq mk_ker),
      map_jacobson_of_surjective hf (le_of_eq mk_ker)] at this
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq I.radical I.jacobson
      this : Eq (Ideal.map (Ideal.Quotient.mk I) I).radical (Ideal.map (Ideal.Quotie …
      ⊢ Eq Bot.bot.radical Bot.bot.jacobson
    -/
    simpa using this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      ⊢ Eq Bot.bot.radical Bot.bot.jacobson → Eq I.radical I.jacobson
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq Bot.bot.radical Bot.bot.jacobson
      ⊢ Eq I.radical I.jacobson
    -/
    have := congr_arg (comap (Ideal.Quotient.mk I)) h
    rw [comap_radical, comap_jacobson_of_surjective hf,
      ← RingHom.ker_eq_comap_bot (Ideal.Quotient.mk I)] at this
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hf : Function.Surjective ⇑(Ideal.Quotient.mk I)
      h : Eq Bot.bot.radical Bot.bot.jacobson
      this : Eq (RingHom.ker (Ideal.Quotient.mk I)).radical (RingHom.ker (Ideal.Quot …
      ⊢ Eq I.radical I.jacobson
    -/
    simpa using this
    /-
      🎉 no goals
    -/


theorem jacobson_radical_eq_jacobson : I.radical.jacobson = I.jacobson :=
  le_antisymm
    (le_trans (le_of_eq (congr_arg jacobson (radical_eq_sInf I)))
      (sInf_le_sInf fun _ hJ => ⟨sInf_le ⟨hJ.1, hJ.2.isPrime⟩, hJ.2⟩))
    (jacobson_mono le_radical)


/-- An ideal `I` is local iff its Jacobson radical is maximal. -/
class IsLocal (I : Ideal R) : Prop where
  /-- A ring `R` is local if and only if its jacobson radical is maximal -/
  out : IsMaximal (jacobson I)


theorem isLocal_iff {I : Ideal R} : IsLocal I ↔ IsMaximal (jacobson I) :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem isLocal_of_isMaximal_radical {I : Ideal R} (hi : IsMaximal (radical I)) : IsLocal I :=
  ⟨have : radical I = jacobson I :=
      le_antisymm (le_sInf fun _ ⟨him, hm⟩ => hm.isPrime.radical_le_iff.2 him)
        (sInf_le ⟨le_radical, hi⟩)
    show IsMaximal (jacobson I) from this ▸ hi⟩


theorem IsLocal.le_jacobson {I J : Ideal R} (hi : IsLocal I) (hij : I ≤ J) (hj : J ≠ ⊤) :
    J ≤ jacobson I :=
  let ⟨_, hm, hjm⟩ := exists_le_maximal J hj
  le_trans hjm <| le_of_eq <| Eq.symm <| hi.1.eq_of_le hm.1.1 <| sInf_le ⟨le_trans hij hjm, hm⟩


theorem IsLocal.mem_jacobson_or_exists_inv {I : Ideal R} (hi : IsLocal I) (x : R) :
    x ∈ jacobson I ∨ ∃ y, y * x - 1 ∈ I :=
  by_cases
    (fun h : I ⊔ span {x} = ⊤ =>
      let ⟨p, hpi, q, hq, hpq⟩ := Submodule.mem_sup.1 ((eq_top_iff_one _).1 h)
      let ⟨r, hr⟩ := mem_span_singleton.1 hq
      Or.inr ⟨r, by
        /-
          R : Type u
          inst✝ : CommRing R
          I : Ideal R
          hi : I.IsLocal
          x : R
          h : Eq (Max.max I (Ideal.span (Singleton.singleton x))) Top.top
          p : R
          hpi : Membership.mem I p
          q : R
          hq : Membership.mem (Ideal.span (Singleton.singleton x)) q
          hpq : Eq (HAdd.hAdd p q) 1
          r : R
          hr : Eq q (HMul.hMul x r)
          ⊢ Membership.mem I (HSub.hSub (HMul.hMul r x) 1)
        -/
        rw [← hpq, mul_comm, ← hr, ← neg_sub, add_sub_cancel_right]; exact I.neg_mem hpi⟩)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    fun h : I ⊔ span {x} ≠ ⊤ =>
    Or.inl <|
      le_trans le_sup_right (hi.le_jacobson le_sup_left h) <| mem_span_singleton.2 <| dvd_refl x


theorem isPrimary_of_isMaximal_radical [CommRing R] {I : Ideal R} (hi : IsMaximal (radical I)) :
    I.IsPrimary :=
  have : radical I = jacobson I :=
    le_antisymm (le_sInf fun _ ⟨him, hm⟩ => hm.isPrime.radical_le_iff.2 him)
      (sInf_le ⟨le_radical, hi⟩)
  isPrimary_iff.mpr
  ⟨ne_top_of_lt <| lt_of_le_of_lt le_radical (lt_top_iff_ne_top.2 hi.1.1), fun {x y} hxy =>
    ((isLocal_of_isMaximal_radical hi).mem_jacobson_or_exists_inv y).symm.imp
      (fun ⟨z, hz⟩ => by
        /-
          R : Type u
          inst✝ : CommRing R
          I : Ideal R
          hi : I.radical.IsMaximal
          this : Eq I.radical I.jacobson
          x y : R
          hxy : Membership.mem I (HMul.hMul x y)
          x✝ : Exists fun y_1 => Membership.mem I (HSub.hSub (HMul.hMul y_1 y) 1)
          z : R
          hz : Membership.mem I (HSub.hSub (HMul.hMul z y) 1)
          ⊢ Membership.mem I x
        -/
        rw [← mul_one x, ← sub_sub_cancel (z * y) 1, mul_sub, mul_left_comm]
        /-
          R : Type u
          inst✝ : CommRing R
          I : Ideal R
          hi : I.radical.IsMaximal
          this : Eq I.radical I.jacobson
          x y : R
          hxy : Membership.mem I (HMul.hMul x y)
          x✝ : Exists fun y_1 => Membership.mem I (HSub.hSub (HMul.hMul y_1 y) 1)
          z : R
          hz : Membership.mem I (HSub.hSub (HMul.hMul z y) 1)
          ⊢ Membership.mem I (HSub.hSub (HMul.hMul z (HMul.hMul x y)) (HMul.hMul x (HSub …
        -/
        exact I.sub_mem (I.mul_mem_left _ hxy) (I.mul_mem_left _ hz))
        /-
          🎉 no goals
        -/
      (this ▸ id)⟩


/-- The Jacobson radical of `I` is the infimum of all maximal (left) ideals containing `I`. -/
def jacobson (I : TwoSidedIdeal R) : TwoSidedIdeal R :=
  (asIdeal I).jacobson.toTwoSided (Ideal.jacobson_mul_mem_right <| I.mul_mem_right _ _)


lemma asIdeal_jacobson (I : TwoSidedIdeal R) : asIdeal I.jacobson = (asIdeal I).jacobson := by
  /-
    R : Type u
    inst✝ : Ring R
    I : TwoSidedIdeal R
    ⊢ Eq (TwoSidedIdeal.asIdeal I.jacobson) (TwoSidedIdeal.asIdeal I).jacobson
  -/
  ext; simp [jacobson]
       /-
         🎉 no goals
       -/


theorem mem_jacobson_iff {x : R} {I : TwoSidedIdeal R} :
    x ∈ jacobson I ↔ ∀ y, ∃ z, z * y * x + z - 1 ∈ I := by
  /-
    R : Type u
    inst✝ : Ring R
    x : R
    I : TwoSidedIdeal R
    ⊢ Iff (Membership.mem I.jacobson x) (∀ (y : R), Exists fun z => Membership.mem …
  -/
  simp [jacobson, Ideal.mem_jacobson_iff]
  /-
    🎉 no goals
  -/


