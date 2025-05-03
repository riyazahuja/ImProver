/-- `rank f` is the rank of a `LinearMap` `f`, defined as the dimension of `f.range`. -/
abbrev rank (f : V →ₗ[K] V') : Cardinal :=
  Module.rank K (LinearMap.range f)


theorem rank_le_range (f : V →ₗ[K] V') : rank f ≤ Module.rank K V' :=
  Submodule.rank_le _


theorem rank_le_domain (f : V →ₗ[K] V₁) : rank f ≤ Module.rank K V :=
  rank_range_le _


@[simp]
theorem rank_zero [Nontrivial K] : rank (0 : V →ₗ[K] V') = 0 := by
  /-
    K : Type u
    V : Type v
    V' : Type v'
    inst✝⁵ : Ring K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    inst✝² : AddCommGroup V'
    inst✝¹ : Module K V'
    inst✝ : Nontrivial K
    ⊢ Eq (LinearMap.rank 0) 0
  -/
  rw [rank, LinearMap.range_zero, rank_bot]
  /-
    🎉 no goals
  -/


theorem rank_comp_le_left (g : V →ₗ[K] V') (f : V' →ₗ[K] V'') : rank (f.comp g) ≤ rank f := by
  /-
    K : Type u
    V : Type v
    V' : Type v'
    V'' : Type v''
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V''
    inst✝ : Module K V''
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V''
    ⊢ LE.le (f.comp g).rank f.rank
  -/
  refine Submodule.rank_mono ?_
  /-
    K : Type u
    V : Type v
    V' : Type v'
    V'' : Type v''
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V''
    inst✝ : Module K V''
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V''
    ⊢ LE.le (LinearMap.range (f.comp g)) (LinearMap.range f)
  -/
  rw [LinearMap.range_comp]
  /-
    K : Type u
    V : Type v
    V' : Type v'
    V'' : Type v''
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V''
    inst✝ : Module K V''
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V''
    ⊢ LE.le (Submodule.map f (LinearMap.range g)) (LinearMap.range f)
  -/
  exact LinearMap.map_le_range
  /-
    🎉 no goals
  -/


theorem lift_rank_comp_le_right (g : V →ₗ[K] V') (f : V' →ₗ[K] V'') :
    Cardinal.lift.{v'} (rank (f.comp g)) ≤ Cardinal.lift.{v''} (rank g) := by
  /-
    K : Type u
    V : Type v
    V' : Type v'
    V'' : Type v''
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V''
    inst✝ : Module K V''
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V''
    ⊢ LE.le (Cardinal.lift.{v', v''} (f.comp g).rank) (Cardinal.lift.{v'', v'} g.r …
  -/
  rw [rank, rank, LinearMap.range_comp]; exact lift_rank_map_le _ _
                                         /-
                                           🎉 no goals
                                         -/


/-- The rank of the composition of two maps is less than the minimum of their ranks. -/
theorem lift_rank_comp_le (g : V →ₗ[K] V') (f : V' →ₗ[K] V'') :
    Cardinal.lift.{v'} (rank (f.comp g)) ≤
      min (Cardinal.lift.{v'} (rank f)) (Cardinal.lift.{v''} (rank g)) :=
  le_min (Cardinal.lift_le.mpr <| rank_comp_le_left _ _) (lift_rank_comp_le_right _ _)


theorem rank_comp_le_right (g : V →ₗ[K] V') (f : V' →ₗ[K] V'₁) : rank (f.comp g) ≤ rank g := by
  /-
    K : Type u
    V : Type v
    V' V'₁ : Type v'
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V'₁
    inst✝ : Module K V'₁
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V'₁
    ⊢ LE.le (f.comp g).rank g.rank
  -/
  simpa only [Cardinal.lift_id] using lift_rank_comp_le_right g f
  /-
    🎉 no goals
  -/


/-- The rank of the composition of two maps is less than the minimum of their ranks.

See `lift_rank_comp_le` for the universe-polymorphic version. -/
theorem rank_comp_le (g : V →ₗ[K] V') (f : V' →ₗ[K] V'₁) :
    rank (f.comp g) ≤ min (rank f) (rank g) := by
  /-
    K : Type u
    V : Type v
    V' V'₁ : Type v'
    inst✝⁶ : Ring K
    inst✝⁵ : AddCommGroup V
    inst✝⁴ : Module K V
    inst✝³ : AddCommGroup V'
    inst✝² : Module K V'
    inst✝¹ : AddCommGroup V'₁
    inst✝ : Module K V'₁
    g : LinearMap (RingHom.id K) V V'
    f : LinearMap (RingHom.id K) V' V'₁
    ⊢ LE.le (f.comp g).rank (Min.min f.rank g.rank)
  -/
  simpa only [Cardinal.lift_id] using lift_rank_comp_le g f
  /-
    🎉 no goals
  -/


theorem rank_add_le (f g : V →ₗ[K] V') : rank (f + g) ≤ rank f + rank g :=
  calc
    rank (f + g) ≤ Module.rank K (LinearMap.range f ⊔ LinearMap.range g : Submodule K V') := by
      /-
        K : Type u
        V : Type v
        V' : Type v'
        inst✝⁴ : DivisionRing K
        inst✝³ : AddCommGroup V
        inst✝² : Module K V
        inst✝¹ : AddCommGroup V'
        inst✝ : Module K V'
        f g : LinearMap (RingHom.id K) V V'
        ⊢ LE.le (HAdd.hAdd f g).rank (Module.rank K (Subtype fun x => Membership.mem ( …
      -/
      refine Submodule.rank_mono ?_
      exact LinearMap.range_le_iff_comap.2 <| eq_top_iff'.2 fun x =>
        show f x + g x ∈ (LinearMap.range f ⊔ LinearMap.range g : Submodule K V') from
        mem_sup.2 ⟨_, ⟨x, rfl⟩, _, ⟨x, rfl⟩, rfl⟩
    _ ≤ rank f + rank g := Submodule.rank_add_le_rank_add_rank _ _


theorem rank_finset_sum_le {η} (s : Finset η) (f : η → V →ₗ[K] V') :
    rank (∑ d ∈ s, f d) ≤ ∑ d ∈ s, rank (f d) :=
  @Finset.sum_hom_rel _ _ _ _ _ (fun a b => rank a ≤ b) f (fun d => rank (f d)) s
    (le_of_eq rank_zero) fun _ _ _ h => le_trans (rank_add_le _ _) (add_le_add_left h _)


theorem le_rank_iff_exists_linearIndependent {c : Cardinal} {f : V →ₗ[K] V'} :
    c ≤ rank f ↔ ∃ s : Set V,
    Cardinal.lift.{v'} #s = Cardinal.lift.{v} c ∧ LinearIndependent K (fun x : s => f x) := by
  /-
    K : Type u
    V : Type v
    V' : Type v'
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V'
    inst✝ : Module K V'
    c : Cardinal.{v'}
    f : LinearMap (RingHom.id K) V V'
    ⊢ Iff (LE.le c f.rank) (Exists fun s => And (Eq (Cardinal.lift.{v', v} (Cardin …
  -/
  rcases f.rangeRestrict.exists_rightInverse_of_surjective f.range_rangeRestrict with ⟨g, hg⟩
  /-
    case intro
    K : Type u
    V : Type v
    V' : Type v'
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V'
    inst✝ : Module K V'
    c : Cardinal.{v'}
    f : LinearMap (RingHom.id K) V V'
    g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
    hg : Eq (f.rangeRestrict.comp g) LinearMap.id
    ⊢ Iff (LE.le c f.rank) (Exists fun s => And (Eq (Cardinal.lift.{v', v} (Cardin …
  -/
  have fg : LeftInverse f.rangeRestrict g := LinearMap.congr_fun hg
  /-
    case intro
    K : Type u
    V : Type v
    V' : Type v'
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V'
    inst✝ : Module K V'
    c : Cardinal.{v'}
    f : LinearMap (RingHom.id K) V V'
    g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
    hg : Eq (f.rangeRestrict.comp g) LinearMap.id
    fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
    ⊢ Iff (LE.le c f.rank) (Exists fun s => And (Eq (Cardinal.lift.{v', v} (Cardin …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case intro.refine_1
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      c : Cardinal.{v'}
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      h : LE.le c f.rank
      ⊢ Exists fun s => And (Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s)) (Cardinal.l …
    -/
  · rcases _root_.le_rank_iff_exists_linearIndependent.1 h with ⟨s, rfl, si⟩
    /-
      case intro.refine_1.intro.intro
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      si : LinearIndependent K Subtype.val
      h : LE.le (Cardinal.mk ↑s) f.rank
      ⊢ Exists fun s_1 => And (Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s_1)) (Cardin …
    -/
    refine ⟨g '' s, Cardinal.mk_image_eq_lift _ _ fg.injective, ?_⟩
    replace fg : ∀ x, f (g x) = x := by
      intro x
      convert congr_arg Subtype.val (fg x)
    replace si : LinearIndependent K fun x : s => f (g x) := by
      simpa only [fg] using si.map' _ (ker_subtype _)
    /-
      case intro.refine_1.intro.intro
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      s : Set (Subtype fun x => Membership.mem (LinearMap.range f) x)
      h : LE.le (Cardinal.mk ↑s) f.rank
      fg : ∀ (x : Subtype fun x => Membership.mem (LinearMap.range f) x), Eq (f (g x …
      si : LinearIndependent K fun x => f (g ↑x)
      ⊢ LinearIndependent K fun x => f ↑x
    -/
    exact si.image_of_comp s g f
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      c : Cardinal.{v'}
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      ⊢ (Exists fun s => And (Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s)) (Cardinal. …
    -/
  · rintro ⟨s, hsc, si⟩
    have : LinearIndependent K fun x : s => f.rangeRestrict x :=
      LinearIndependent.of_comp f.range.subtype (by convert si)
    /-
      case intro.refine_2.intro.intro
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      c : Cardinal.{v'}
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      s : Set V
      hsc : Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s)) (Cardinal.lift.{v, v'} c)
      si : LinearIndependent K fun x => f ↑x
      this : LinearIndependent K fun x => f.rangeRestrict ↑x
      ⊢ LE.le c f.rank
    -/
    convert this.image.cardinal_le_rank
    /-
      case h.e'_3
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      c : Cardinal.{v'}
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      s : Set V
      hsc : Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s)) (Cardinal.lift.{v, v'} c)
      si : LinearIndependent K fun x => f ↑x
      this : LinearIndependent K fun x => f.rangeRestrict ↑x
      ⊢ Eq c (Cardinal.mk ↑(Set.image (⇑f.rangeRestrict) s))
    -/
    rw [← Cardinal.lift_inj, ← hsc, Cardinal.mk_image_eq_of_injOn_lift]
    /-
      case h.e'_3.h
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      c : Cardinal.{v'}
      f : LinearMap (RingHom.id K) V V'
      g : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LinearMap.range …
      hg : Eq (f.rangeRestrict.comp g) LinearMap.id
      fg : Function.LeftInverse ⇑f.rangeRestrict ⇑g
      s : Set V
      hsc : Eq (Cardinal.lift.{v', v} (Cardinal.mk ↑s)) (Cardinal.lift.{v, v'} c)
      si : LinearIndependent K fun x => f ↑x
      this : LinearIndependent K fun x => f.rangeRestrict ↑x
      ⊢ Set.InjOn (⇑f.rangeRestrict) s
    -/
    exact injOn_iff_injective.2 this.injective
    /-
      🎉 no goals
    -/


theorem le_rank_iff_exists_linearIndependent_finset {n : ℕ} {f : V →ₗ[K] V'} :
    ↑n ≤ rank f ↔ ∃ s : Finset V, s.card = n ∧ LinearIndependent K fun x : (s : Set V) => f x := by
  simp only [le_rank_iff_exists_linearIndependent, Cardinal.lift_natCast, Cardinal.lift_eq_nat_iff,
    Cardinal.mk_set_eq_nat_iff_finset]
  /-
    K : Type u
    V : Type v
    V' : Type v'
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : AddCommGroup V'
    inst✝ : Module K V'
    n : Nat
    f : LinearMap (RingHom.id K) V V'
    ⊢ Iff (Exists fun s => And (Exists fun t => And (Eq (↑t) s) (Eq t.card n)) (Li …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      n : Nat
      f : LinearMap (RingHom.id K) V V'
      ⊢ (Exists fun s => And (Exists fun t => And (Eq (↑t) s) (Eq t.card n)) (Linear …
    -/
  · rintro ⟨s, ⟨t, rfl, rfl⟩, si⟩
    /-
      case mp.intro.intro.intro.intro
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      f : LinearMap (RingHom.id K) V V'
      t : Finset V
      si : LinearIndependent K fun x => f ↑x
      ⊢ Exists fun s => And (Eq s.card t.card) (LinearIndependent K fun x => f ↑x)
    -/
    exact ⟨t, rfl, si⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      n : Nat
      f : LinearMap (RingHom.id K) V V'
      ⊢ (Exists fun s => And (Eq s.card n) (LinearIndependent K fun x => f ↑x)) → Ex …
    -/
  · rintro ⟨s, rfl, si⟩
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      V' : Type v'
      inst✝⁴ : DivisionRing K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : AddCommGroup V'
      inst✝ : Module K V'
      f : LinearMap (RingHom.id K) V V'
      s : Finset V
      si : LinearIndependent K fun x => f ↑x
      ⊢ Exists fun s_1 => And (Exists fun t => And (Eq (↑t) s_1) (Eq t.card s.card)) …
    -/
    exact ⟨s, ⟨s, rfl, rfl⟩, si⟩
    /-
      🎉 no goals
    -/


