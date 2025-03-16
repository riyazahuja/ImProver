@[to_additive]
theorem card_dvd_exponent_pow_rank : Nat.card G ∣ Monoid.exponent G ^ Group.rank G := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    ⊢ Dvd.dvd (Nat.card G) (HPow.hPow (Monoid.exponent G) (Group.rank G))
  -/
  obtain ⟨S, hS1, hS2⟩ := Group.rank_spec G
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Dvd.dvd (Nat.card G) (HPow.hPow (Monoid.exponent G) (Group.rank G))
  -/
  rw [← hS1, ← Fintype.card_coe, ← Finset.card_univ, ← Finset.prod_const]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Dvd.dvd (Nat.card G) (Finset.univ.prod fun _x => Monoid.exponent G)
  -/
  let f : (∀ g : S, zpowers (g : G)) →* G := noncommPiCoprod fun s t _ x y _ _ => mul_comm x _
  have hf : Function.Surjective f := by
    rw [← MonoidHom.range_eq_top, eq_top_iff, ← hS2, closure_le]
    exact fun g hg => ⟨Pi.mulSingle ⟨g, hg⟩ ⟨g, mem_zpowers g⟩, noncommPiCoprod_mulSingle _ _⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    f : MonoidHom ((g : Subtype fun x => Membership.mem S x) → Subtype fun x => Me …
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Nat.card G) (Finset.univ.prod fun _x => Monoid.exponent G)
  -/
  replace hf := card_dvd_of_surjective f hf
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    f : MonoidHom ((g : Subtype fun x => Membership.mem S x) → Subtype fun x => Me …
    hf : Dvd.dvd (Nat.card G) (Nat.card ((g : Subtype fun x => Membership.mem S x) …
    ⊢ Dvd.dvd (Nat.card G) (Finset.univ.prod fun _x => Monoid.exponent G)
  -/
  rw [Nat.card_pi] at hf
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    f : MonoidHom ((g : Subtype fun x => Membership.mem S x) → Subtype fun x => Me …
    hf : Dvd.dvd (Nat.card G) (Finset.univ.prod fun a => Nat.card (Subtype fun x = …
    ⊢ Dvd.dvd (Nat.card G) (Finset.univ.prod fun _x => Monoid.exponent G)
  -/
  refine hf.trans (Finset.prod_dvd_prod_of_dvd _ _ fun g _ => ?_)
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    f : MonoidHom ((g : Subtype fun x => Membership.mem S x) → Subtype fun x => Me …
    hf : Dvd.dvd (Nat.card G) (Finset.univ.prod fun a => Nat.card (Subtype fun x = …
    g : Subtype fun x => Membership.mem S x
    x✝ : Membership.mem Finset.univ g
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (Subgroup.zpowers ↑g) x)) …
  -/
  rw [Nat.card_zpowers]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Group.FG G
    S : Finset G
    hS1 : Eq S.card (Group.rank G)
    hS2 : Eq (Subgroup.closure ↑S) Top.top
    f : MonoidHom ((g : Subtype fun x => Membership.mem S x) → Subtype fun x => Me …
    hf : Dvd.dvd (Nat.card G) (Finset.univ.prod fun a => Nat.card (Subtype fun x = …
    g : Subtype fun x => Membership.mem S x
    x✝ : Membership.mem Finset.univ g
    ⊢ Dvd.dvd (orderOf ↑g) (Monoid.exponent G)
  -/
  exact Monoid.order_dvd_exponent (g : G)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem card_dvd_exponent_pow_rank' {n : ℕ} (hG : ∀ g : G, g ^ n = 1) :
    Nat.card G ∣ n ^ Group.rank G :=
  (card_dvd_exponent_pow_rank G).trans
    (pow_dvd_pow_of_dvd (Monoid.exponent_dvd_of_forall_pow_eq_one hG) (Group.rank G))


theorem closure_mul_image_mul_eq_top
    (hR : IsComplement H R) (hR1 : (1 : G) ∈ R) (hS : closure S = ⊤) :
    (closure ((R * S).image fun g => g * (hR.toRightFun g : G)⁻¹)) * R = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    ⊢ Eq (HMul.hMul (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv  …
  -/
  let f : G → R := hR.toRightFun
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    f : G → ↑R := hR.toRightFun
    ⊢ Eq (HMul.hMul (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv  …
  -/
  let U : Set G := (R * S).image fun g => g * (f g : G)⁻¹
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    f : G → ↑R := hR.toRightFun
    U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
    ⊢ Eq (HMul.hMul (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv  …
  -/
  change (closure U : Set G) * R = ⊤
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    f : G → ↑R := hR.toRightFun
    U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
    ⊢ Eq (HMul.hMul (↑(Subgroup.closure U)) R) Top.top
  -/
  refine top_le_iff.mp fun g _ => ?_
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    f : G → ↑R := hR.toRightFun
    U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
    g : G
    x✝ : Membership.mem Top.top g
    ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) g
  -/
  refine closure_induction_right ?_ ?_ ?_ (eq_top_iff.mp hS (mem_top g))
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) 1
    -/
  · exact ⟨1, (closure U).one_mem, 1, hR1, one_mul 1⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      ⊢ ∀ (x : G), Membership.mem (Subgroup.closure S) x → ∀ (y : G), Membership.mem …
    -/
  · rintro - - s hs ⟨u, hu, r, hr, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) (HMul.hMul ((fun x1 x2  …
    -/
    rw [show u * r * s = u * (r * s * (f (r * s) : G)⁻¹) * f (r * s) by group]
    /-
      case refine_2.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) (HMul.hMul (HMul.hMul u …
    -/
    refine Set.mul_mem_mul ((closure U).mul_mem hu ?_) (f (r * s)).coe_prop
    /-
      case refine_2.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (Subgroup.closure U) (HMul.hMul (HMul.hMul r s) (Inv.inv ↑(f  …
    -/
    exact subset_closure ⟨r * s, Set.mul_mem_mul hr hs, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      ⊢ ∀ (x : G), Membership.mem (Subgroup.closure S) x → ∀ (y : G), Membership.mem …
    -/
  · rintro - - s hs ⟨u, hu, r, hr, rfl⟩
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) (HMul.hMul ((fun x1 x2  …
    -/
    rw [show u * r * s⁻¹ = u * (f (r * s⁻¹) * s * r⁻¹)⁻¹ * f (r * s⁻¹) by group]
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (HMul.hMul (↑(Subgroup.closure U)) R) (HMul.hMul (HMul.hMul u …
    -/
    refine Set.mul_mem_mul ((closure U).mul_mem hu ((closure U).inv_mem ?_)) (f (r * s⁻¹)).2
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (Subgroup.closure U) (HMul.hMul (HMul.hMul (↑(f (HMul.hMul r  …
    -/
    refine subset_closure ⟨f (r * s⁻¹) * s, Set.mul_mem_mul (f (r * s⁻¹)).2 hs, ?_⟩
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Eq ((fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul (↑(f (HMul.hMul r (In …
    -/
    rw [mul_right_inj, inv_inj, ← Subtype.coe_mk r hr, ← Subtype.ext_iff, Subtype.coe_mk]
    apply (isComplement_iff_existsUnique_mul_inv_mem.mp hR (f (r * s⁻¹) * s)).unique
      (hR.mul_inv_toRightFun_mem (f (r * s⁻¹) * s))
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (↑H) (HMul.hMul (HMul.hMul (↑(f (HMul.hMul r (Inv.inv s)))) s …
    -/
    rw [mul_assoc, ← inv_inv s, ← mul_inv_rev, inv_inv]
    /-
      case refine_3.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      f : G → ↑R := hR.toRightFun
      U : Set G := Set.image (fun g => HMul.hMul g (Inv.inv ↑(f g))) (HMul.hMul R S)
      g : G
      x✝ : Membership.mem Top.top g
      s : G
      hs : Membership.mem S s
      u : G
      hu : Membership.mem (↑(Subgroup.closure U)) u
      r : G
      hr : Membership.mem R r
      ⊢ Membership.mem (↑H) (HMul.hMul (↑(f (HMul.hMul r (Inv.inv s)))) (Inv.inv (HM …
    -/
    exact hR.toRightFun_mul_inv_mem (r * s⁻¹)
    /-
      🎉 no goals
    -/


/-- **Schreier's Lemma**: If `R : Set G` is a `rightTransversal` of `H : Subgroup G`
  with `1 ∈ R`, and if `G` is generated by `S : Set G`, then `H` is generated by the `Set`
  `(R * S).image (fun g ↦ g * (hR.toRightFun g)⁻¹)`. -/
theorem closure_mul_image_eq (hR : IsComplement H R) (hR1 : (1 : G) ∈ R)
    (hS : closure S = ⊤) : closure ((R * S).image fun g => g * (hR.toRightFun g : G)⁻¹) = H := by
  have hU : closure ((R * S).image fun g => g * (hR.toRightFun g : G)⁻¹) ≤ H := by
    rw [closure_le]
    rintro - ⟨g, -, rfl⟩
    exact hR.mul_inv_toRightFun_mem g
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
    ⊢ Eq (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.toRightF …
  -/
  refine le_antisymm hU fun h hh => ?_
  obtain ⟨g, hg, r, hr, rfl⟩ :=
    show h ∈ _ from eq_top_iff.mp (closure_mul_image_mul_eq_top hR hR1 hS) (mem_top h)
  suffices (⟨r, hr⟩ : R) = (⟨1, hR1⟩ : R) by
    simpa only [show r = 1 from Subtype.ext_iff.mp this, mul_one]
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
    g : G
    hg : Membership.mem (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv. …
    r : G
    hr : Membership.mem R r
    hh : Membership.mem H ((fun x1 x2 => HMul.hMul x1 x2) g r)
    ⊢ Eq ⟨r, hr⟩ ⟨1, hR1⟩
  -/
  apply (isComplement_iff_existsUnique_mul_inv_mem.mp hR r).unique
    /-
      case intro.intro.intro.intro.py₁
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
      g : G
      hg : Membership.mem (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv. …
      r : G
      hr : Membership.mem R r
      hh : Membership.mem H ((fun x1 x2 => HMul.hMul x1 x2) g r)
      ⊢ Membership.mem (↑H) (HMul.hMul r (Inv.inv ↑⟨r, hr⟩))
    -/
  · rw [Subtype.coe_mk, mul_inv_cancel]
    /-
      case intro.intro.intro.intro.py₁
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
      g : G
      hg : Membership.mem (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv. …
      r : G
      hr : Membership.mem R r
      hh : Membership.mem H ((fun x1 x2 => HMul.hMul x1 x2) g r)
      ⊢ Membership.mem (↑H) 1
    -/
    exact H.one_mem
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.py₂
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
      g : G
      hg : Membership.mem (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv. …
      r : G
      hr : Membership.mem R r
      hh : Membership.mem H ((fun x1 x2 => HMul.hMul x1 x2) g r)
      ⊢ Membership.mem (↑H) (HMul.hMul r (Inv.inv ↑⟨1, hR1⟩))
    -/
  · rw [Subtype.coe_mk, inv_one, mul_one]
    /-
      case intro.intro.intro.intro.py₂
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      R S : Set G
      hR : Subgroup.IsComplement (↑H) R
      hR1 : Membership.mem R 1
      hS : Eq (Subgroup.closure S) Top.top
      hU : LE.le (Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv.inv ↑(hR.to …
      g : G
      hg : Membership.mem (↑(Subgroup.closure (Set.image (fun g => HMul.hMul g (Inv. …
      r : G
      hr : Membership.mem R r
      hh : Membership.mem H ((fun x1 x2 => HMul.hMul x1 x2) g r)
      ⊢ Membership.mem (↑H) r
    -/
    exact (H.mul_mem_cancel_left (hU hg)).mp hh
    /-
      🎉 no goals
    -/


/-- **Schreier's Lemma**: If `R : Set G` is a `rightTransversal` of `H : Subgroup G`
  with `1 ∈ R`, and if `G` is generated by `S : Set G`, then `H` is generated by the `Set`
  `(R * S).image (fun g ↦ g * (hR.toRightFun g)⁻¹)`. -/
theorem closure_mul_image_eq_top (hR : IsComplement H R) (hR1 : (1 : G) ∈ R)
    (hS : closure S = ⊤) : closure ((R * S).image fun g =>
      ⟨g * (hR.toRightFun g : G)⁻¹, hR.mul_inv_toRightFun_mem g⟩ : Set H) = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    ⊢ Eq (Subgroup.closure (Set.image (fun g => ⟨HMul.hMul g (Inv.inv ↑(hR.toRight …
  -/
  rw [eq_top_iff, ← map_subtype_le_map_subtype, MonoidHom.map_closure, Set.image_image]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    R S : Set G
    hR : Subgroup.IsComplement (↑H) R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure S) Top.top
    ⊢ LE.le (Subgroup.map H.subtype Top.top) (Subgroup.closure (Set.image (fun x = …
  -/
  exact (map_subtype_le ⊤).trans (ge_of_eq (closure_mul_image_eq hR hR1 hS))
  /-
    🎉 no goals
  -/


/-- **Schreier's Lemma**: If `R : Finset G` is a `rightTransversal` of `H : Subgroup G`
  with `1 ∈ R`, and if `G` is generated by `S : Finset G`, then `H` is generated by the `Finset`
  `(R * S).image (fun g ↦ g * (hR.toRightFun g)⁻¹)`. -/
theorem closure_mul_image_eq_top' [DecidableEq G] {R S : Finset G}
    (hR : IsComplement (H : Set G) R) (hR1 : (1 : G) ∈ R)
    (hS : closure (S : Set G) = ⊤) :
    closure (((R * S).image fun g => ⟨_, hR.mul_inv_toRightFun_mem g⟩ : Finset H) : Set H) = ⊤ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : DecidableEq G
    R S : Finset G
    hR : Subgroup.IsComplement ↑H ↑R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Eq (Subgroup.closure ↑(Finset.image (fun g => ⟨HMul.hMul g (Inv.inv ↑(hR.toR …
  -/
  rw [Finset.coe_image, Finset.coe_mul]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : DecidableEq G
    R S : Finset G
    hR : Subgroup.IsComplement ↑H ↑R
    hR1 : Membership.mem R 1
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Eq (Subgroup.closure (Set.image (fun g => ⟨HMul.hMul g (Inv.inv ↑(hR.toRight …
  -/
  exact closure_mul_image_eq_top hR hR1 hS
  /-
    🎉 no goals
  -/


theorem exists_finset_card_le_mul [FiniteIndex H] {S : Finset G} (hS : closure (S : Set G) = ⊤) :
    ∃ T : Finset H, T.card ≤ H.index * S.card ∧ closure (T : Set H) = ⊤ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  letI := H.fintypeQuotientOfFiniteIndex
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  haveI : DecidableEq G := Classical.decEq G
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this : DecidableEq G
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  obtain ⟨R₀, hR, hR1⟩ := H.exists_isComplement_right 1
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this : DecidableEq G
    R₀ : Set G
    hR : Subgroup.IsComplement (↑H) R₀
    hR1 : Membership.mem R₀ 1
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  haveI : Fintype R₀ := Fintype.ofEquiv _ hR.rightQuotientEquiv
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝¹ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this✝ : DecidableEq G
    R₀ : Set G
    hR : Subgroup.IsComplement (↑H) R₀
    hR1 : Membership.mem R₀ 1
    this : Fintype ↑R₀
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  let R : Finset G := Set.toFinset R₀
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝¹ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this✝ : DecidableEq G
    R₀ : Set G
    hR : Subgroup.IsComplement (↑H) R₀
    hR1 : Membership.mem R₀ 1
    this : Fintype ↑R₀
    R : Finset G := R₀.toFinset
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  replace hR : IsComplement (H : Set G) R := by rwa [Set.coe_toFinset]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝¹ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this✝ : DecidableEq G
    R₀ : Set G
    hR1 : Membership.mem R₀ 1
    this : Fintype ↑R₀
    R : Finset G := R₀.toFinset
    hR : Subgroup.IsComplement ↑H ↑R
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  replace hR1 : (1 : G) ∈ R := by rwa [Set.mem_toFinset]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    this✝¹ : Fintype (HasQuotient.Quotient G H) := H.fintypeQuotientOfFiniteIndex
    this✝ : DecidableEq G
    R₀ : Set G
    this : Fintype ↑R₀
    R : Finset G := R₀.toFinset
    hR : Subgroup.IsComplement ↑H ↑R
    hR1 : Membership.mem R 1
    ⊢ Exists fun T => And (LE.le T.card (HMul.hMul H.index S.card)) (Eq (Subgroup. …
  -/
  refine ⟨_, ?_, closure_mul_image_eq_top' hR hR1 hS⟩
  calc
    _ ≤ (R * S).card := Finset.card_image_le
    _ ≤ (R ×ˢ S).card := Finset.card_image_le
    _ = R.card * S.card := R.card_product S
    _ = H.index * S.card := congr_arg (· * S.card) ?_
  calc
    R.card = Fintype.card R := (Fintype.card_coe R).symm
    _ = _ := (Fintype.card_congr hR.rightQuotientEquiv).symm
    _ = Fintype.card (G ⧸ H) := QuotientGroup.card_quotient_rightRel H
    _ = H.index := by rw [index_eq_card, Nat.card_eq_fintype_card]


/-- **Schreier's Lemma**: A finite index subgroup of a finitely generated
  group is finitely generated. -/
instance fg_of_index_ne_zero [hG : Group.FG G] [FiniteIndex H] : Group.FG H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S : Set G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    ⊢ Group.FG (Subtype fun x => Membership.mem H x)
  -/
  obtain ⟨S, hS⟩ := hG.1
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S✝ : Set G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ Group.FG (Subtype fun x => Membership.mem H x)
  -/
  obtain ⟨T, -, hT⟩ := exists_finset_card_le_mul H hS
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S✝ : Set G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    S : Finset G
    hS : Eq (Subgroup.closure ↑S) Top.top
    T : Finset (Subtype fun x => Membership.mem H x)
    hT : Eq (Subgroup.closure ↑T) Top.top
    ⊢ Group.FG (Subtype fun x => Membership.mem H x)
  -/
  exact ⟨⟨T, hT⟩⟩
  /-
    🎉 no goals
  -/


theorem rank_le_index_mul_rank [hG : Group.FG G] [FiniteIndex H] :
    Group.rank H ≤ H.index * Group.rank G := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    ⊢ LE.le (Group.rank (Subtype fun x => Membership.mem H x)) (HMul.hMul H.index  …
  -/
  haveI := H.fg_of_index_ne_zero
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    this : Group.FG (Subtype fun x => Membership.mem H x)
    ⊢ LE.le (Group.rank (Subtype fun x => Membership.mem H x)) (HMul.hMul H.index  …
  -/
  obtain ⟨S, hS₀, hS⟩ := Group.rank_spec G
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    hG : Group.FG G
    inst✝ : H.FiniteIndex
    this : Group.FG (Subtype fun x => Membership.mem H x)
    S : Finset G
    hS₀ : Eq S.card (Group.rank G)
    hS : Eq (Subgroup.closure ↑S) Top.top
    ⊢ LE.le (Group.rank (Subtype fun x => Membership.mem H x)) (HMul.hMul H.index  …
  -/
  obtain ⟨T, hT₀, hT⟩ := exists_finset_card_le_mul H hS
  calc
    Group.rank H ≤ T.card := Group.rank_le H hT
    _ ≤ H.index * S.card := hT₀
    _ = H.index * Group.rank G := congr_arg (H.index * ·) hS₀


/-- If `G` has `n` commutators `[g₁, g₂]`, then `|G'| ∣ [G : Z(G)] ^ ([G : Z(G)] * n + 1)`,
where `G'` denotes the commutator of `G`. -/
theorem card_commutator_dvd_index_center_pow [Finite (commutatorSet G)] :
    Nat.card (_root_.commutator G) ∣
      (center G).index ^ ((center G).index * Nat.card (commutatorSet G) + 1) := by
  -- First handle the case when `Z(G)` has infinite index and `[G : Z(G)]` is defined to be `0`
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) …
  -/
  by_cases hG : (center G).index = 0
    /-
      case pos
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Finite ↑(commutatorSet G)
      hG : Eq (Subgroup.center G).index 0
      ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) …
    -/
  · simp_rw [hG, zero_mul, zero_add, pow_one, dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) …
  -/
  haveI : FiniteIndex (center G) := ⟨hG⟩
  -- Rewrite as `|Z(G) ∩ G'| * [G' : Z(G) ∩ G'] ∣ [G : Z(G)] ^ ([G : Z(G)] * n) * [G : Z(G)]`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this : (Subgroup.center G).FiniteIndex
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) …
  -/
  rw [← ((center G).subgroupOf (_root_.commutator G)).card_mul_index, pow_succ]
  -- We have `h1 : [G' : Z(G) ∩ G'] ∣ [G : Z(G)]`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this : (Subgroup.center G).FiniteIndex
    ⊢ Dvd.dvd (HMul.hMul (Nat.card (Subtype fun x => Membership.mem ((Subgroup.cen …
  -/
  have h1 := relindex_dvd_index_of_normal (center G) (_root_.commutator G)
  -- So we can reduce to proving `|Z(G) ∩ G'| ∣ [G : Z(G)] ^ ([G : Z(G)] * n)`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    ⊢ Dvd.dvd (HMul.hMul (Nat.card (Subtype fun x => Membership.mem ((Subgroup.cen …
  -/
  refine mul_dvd_mul ?_ h1
  -- We know that `[G' : Z(G) ∩ G'] < ∞` by `h1` and `hG`
  haveI : FiniteIndex ((center G).subgroupOf (_root_.commutator G)) :=
    ⟨ne_zero_of_dvd_ne_zero hG h1⟩
  -- We have `h2 : rank (Z(G) ∩ G') ≤ [G' : Z(G) ∩ G'] * rank G'` by Schreier's lemma
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem ((Subgroup.center G).subg …
  -/
  have h2 := rank_le_index_mul_rank ((center G).subgroupOf (_root_.commutator G))
  -- We have `h3 : [G' : Z(G) ∩ G'] * rank G' ≤ [G : Z(G)] * n` by `h1` and `rank G' ≤ n`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem ((Subgroup.center G).subg …
  -/
  have h3 := Nat.mul_le_mul (Nat.le_of_dvd (Nat.pos_of_ne_zero hG) h1) (rank_commutator_le_card G)
  -- So we can reduce to proving `|Z(G) ∩ G'| ∣ [G : Z(G)] ^ rank (Z(G) ∩ G')`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    h3 : LE.le (HMul.hMul ((Subgroup.center G).relindex (_root_.commutator G)) (Gr …
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem ((Subgroup.center G).subg …
  -/
  refine dvd_trans ?_ (pow_dvd_pow (center G).index (h2.trans h3))
  -- `Z(G) ∩ G'` is abelian, so it enough to prove that `g ^ [G : Z(G)] = 1` for `g ∈ Z(G) ∩ G'`
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    h3 : LE.le (HMul.hMul ((Subgroup.center G).relindex (_root_.commutator G)) (Gr …
    ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem ((Subgroup.center G).subg …
  -/
  apply card_dvd_exponent_pow_rank'
  /-
    case neg.hG
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    h3 : LE.le (HMul.hMul ((Subgroup.center G).relindex (_root_.commutator G)) (Gr …
    ⊢ ∀ (g : Subtype fun x => Membership.mem ((Subgroup.center G).subgroupOf (_roo …
  -/
  intro g
  -- `Z(G)` is abelian, so `g ∈ Z(G) ∩ G' ≤ G' ≤ ker (transfer : G → Z(G))`
  /-
    case neg.hG
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    h3 : LE.le (HMul.hMul ((Subgroup.center G).relindex (_root_.commutator G)) (Gr …
    g : Subtype fun x => Membership.mem ((Subgroup.center G).subgroupOf (_root_.co …
    ⊢ Eq (HPow.hPow g (Subgroup.center G).index) 1
  -/
  have := Abelianization.commutator_subset_ker (MonoidHom.transferCenterPow G) g.1.2
  -- `transfer g` is defeq to `g ^ [G : Z(G)]`, so we are done
  /-
    case neg.hG
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    hG : Not (Eq (Subgroup.center G).index 0)
    this✝¹ : (Subgroup.center G).FiniteIndex
    h1 : Dvd.dvd ((Subgroup.center G).relindex (_root_.commutator G)) (Subgroup.ce …
    this✝ : ((Subgroup.center G).subgroupOf (_root_.commutator G)).FiniteIndex
    h2 : LE.le (Group.rank (Subtype fun x => Membership.mem ((Subgroup.center G).s …
    h3 : LE.le (HMul.hMul ((Subgroup.center G).relindex (_root_.commutator G)) (Gr …
    g : Subtype fun x => Membership.mem ((Subgroup.center G).subgroupOf (_root_.co …
    this : Membership.mem (MonoidHom.transferCenterPow G).ker ↑↑g
    ⊢ Eq (HPow.hPow g (Subgroup.center G).index) 1
  -/
  simpa only [MonoidHom.mem_ker, Subtype.ext_iff] using this
  /-
    🎉 no goals
  -/


/-- A bound for the size of the commutator subgroup in terms of the number of commutators. -/
def cardCommutatorBound (n : ℕ) :=
  (n ^ (2 * n)) ^ (n ^ (2 * n + 1) + 1)


/-- A theorem of Schur: The size of the commutator subgroup is bounded in terms of the number of
  commutators. -/
theorem card_commutator_le_of_finite_commutatorSet [Finite (commutatorSet G)] :
    Nat.card (_root_.commutator G) ≤ cardCommutatorBound (Nat.card (commutatorSet G)) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  have h1 := index_center_le_pow (closureCommutatorRepresentatives G)
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  have h2 := card_commutator_dvd_index_center_pow (closureCommutatorRepresentatives G)
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator (Su …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  rw [card_commutatorSet_closureCommutatorRepresentatives] at h1 h2
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator (Su …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  rw [card_commutator_closureCommutatorRepresentatives] at h2
  replace h1 :=
    h1.trans
      (Nat.pow_le_pow_of_le_right Finite.card_pos (rank_closureCommutatorRepresentatives_le G))
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G)  …
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  replace h2 := h2.trans (pow_dvd_pow _ (add_le_add_right (mul_le_mul_right' h1 _) 1))
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G)  …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  rw [← pow_succ] at h2
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G)  …
    ⊢ LE.le (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) ( …
  -/
  refine (Nat.le_of_dvd ?_ h2).trans (Nat.pow_le_pow_left h1 _)
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite ↑(commutatorSet G)
    h1 : LE.le (Subgroup.center (Subtype fun x => Membership.mem (closureCommutato …
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G)  …
    ⊢ LT.lt 0 (HPow.hPow (Subgroup.center (Subtype fun x => Membership.mem (closur …
  -/
  exact pow_pos (Nat.pos_of_ne_zero FiniteIndex.finiteIndex) _
  /-
    🎉 no goals
  -/


/-- A theorem of Schur: A group with finitely many commutators has finite commutator subgroup. -/
instance [Finite (commutatorSet G)] : Finite (_root_.commutator G) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S : Set G
    inst✝ : Finite ↑(commutatorSet G)
    ⊢ Finite (Subtype fun x => Membership.mem (_root_.commutator G) x)
  -/
  have h2 := card_commutator_dvd_index_center_pow (closureCommutatorRepresentatives G)
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S : Set G
    inst✝ : Finite ↑(commutatorSet G)
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator (Su …
    ⊢ Finite (Subtype fun x => Membership.mem (_root_.commutator G) x)
  -/
  refine Nat.finite_of_card_ne_zero fun h => ?_
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S : Set G
    inst✝ : Finite ↑(commutatorSet G)
    h2 : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (_root_.commutator (Su …
    h : Eq (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) 0
    ⊢ False
  -/
  rw [card_commutator_closureCommutatorRepresentatives, h, zero_dvd_iff] at h2
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    R S : Set G
    inst✝ : Finite ↑(commutatorSet G)
    h2 : Eq (HPow.hPow (Subgroup.center (Subtype fun x => Membership.mem (closureC …
    h : Eq (Nat.card (Subtype fun x => Membership.mem (_root_.commutator G) x)) 0
    ⊢ False
  -/
  exact FiniteIndex.finiteIndex (pow_eq_zero h2)
  /-
    🎉 no goals
  -/


