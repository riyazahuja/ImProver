theorem Ici_add_bij : BijOn (· + d) (Ici a) (Ici (a + d)) := by
  refine
    ⟨fun x h => add_le_add_right (mem_Ici.mp h) _, (add_left_injective d).injOn, fun _ h => ?_⟩
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d x✝ : M
    h : Membership.mem (Set.Ici (HAdd.hAdd a d)) x✝
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ici a)) x✝
  -/
  obtain ⟨c, rfl⟩ := exists_add_of_le (mem_Ici.mp h)
  /-
    case intro
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d c : M
    h : Membership.mem (Set.Ici (HAdd.hAdd a d)) (HAdd.hAdd (HAdd.hAdd a d) c)
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ici a)) (HAdd.hAdd ( …
  -/
  rw [mem_Ici, add_right_comm, add_le_add_iff_right] at h
  /-
    case intro
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d c : M
    h : LE.le a (HAdd.hAdd a c)
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ici a)) (HAdd.hAdd ( …
  -/
  exact ⟨a + c, h, by rw [add_right_comm]⟩
  /-
    🎉 no goals
  -/


theorem Ioi_add_bij : BijOn (· + d) (Ioi a) (Ioi (a + d)) := by
  refine
    ⟨fun x h => add_lt_add_right (mem_Ioi.mp h) _, fun _ _ _ _ h => add_right_cancel h, fun _ h =>
      ?_⟩
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d x✝ : M
    h : Membership.mem (Set.Ioi (HAdd.hAdd a d)) x✝
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ioi a)) x✝
  -/
  obtain ⟨c, rfl⟩ := exists_add_of_le (mem_Ioi.mp h).le
  /-
    case intro
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d c : M
    h : Membership.mem (Set.Ioi (HAdd.hAdd a d)) (HAdd.hAdd (HAdd.hAdd a d) c)
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ioi a)) (HAdd.hAdd ( …
  -/
  rw [mem_Ioi, add_right_comm, add_lt_add_iff_right] at h
  /-
    case intro
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a d c : M
    h : LT.lt a (HAdd.hAdd a c)
    ⊢ Membership.mem (Set.image (fun x => HAdd.hAdd x d) (Set.Ioi a)) (HAdd.hAdd ( …
  -/
  exact ⟨a + c, h, by rw [add_right_comm]⟩
  /-
    🎉 no goals
  -/


theorem Icc_add_bij : BijOn (· + d) (Icc a b) (Icc (a + d) (b + d)) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b d : M
    ⊢ Set.BijOn (fun x => HAdd.hAdd x d) (Set.Icc a b) (Set.Icc (HAdd.hAdd a d) (H …
  -/
  rw [← Ici_inter_Iic, ← Ici_inter_Iic]
  exact
    (Ici_add_bij a d).inter_mapsTo (fun x hx => add_le_add_right hx _) fun x hx =>
      le_of_add_le_add_right hx.2


theorem Ioo_add_bij : BijOn (· + d) (Ioo a b) (Ioo (a + d) (b + d)) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b d : M
    ⊢ Set.BijOn (fun x => HAdd.hAdd x d) (Set.Ioo a b) (Set.Ioo (HAdd.hAdd a d) (H …
  -/
  rw [← Ioi_inter_Iio, ← Ioi_inter_Iio]
  exact
    (Ioi_add_bij a d).inter_mapsTo (fun x hx => add_lt_add_right hx _) fun x hx =>
      lt_of_add_lt_add_right hx.2


theorem Ioc_add_bij : BijOn (· + d) (Ioc a b) (Ioc (a + d) (b + d)) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b d : M
    ⊢ Set.BijOn (fun x => HAdd.hAdd x d) (Set.Ioc a b) (Set.Ioc (HAdd.hAdd a d) (H …
  -/
  rw [← Ioi_inter_Iic, ← Ioi_inter_Iic]
  exact
    (Ioi_add_bij a d).inter_mapsTo (fun x hx => add_le_add_right hx _) fun x hx =>
      le_of_add_le_add_right hx.2


theorem Ico_add_bij : BijOn (· + d) (Ico a b) (Ico (a + d) (b + d)) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b d : M
    ⊢ Set.BijOn (fun x => HAdd.hAdd x d) (Set.Ico a b) (Set.Ico (HAdd.hAdd a d) (H …
  -/
  rw [← Ici_inter_Iio, ← Ici_inter_Iio]
  exact
    (Ici_add_bij a d).inter_mapsTo (fun x hx => add_lt_add_right hx _) fun x hx =>
      lt_of_add_lt_add_right hx.2


@[simp]
theorem image_add_const_Ici : (fun x => x + a) '' Ici b = Ici (b + a) :=
  (Ici_add_bij _ _).image_eq


@[simp]
theorem image_add_const_Ioi : (fun x => x + a) '' Ioi b = Ioi (b + a) :=
  (Ioi_add_bij _ _).image_eq


@[simp]
theorem image_add_const_Icc : (fun x => x + a) '' Icc b c = Icc (b + a) (c + a) :=
  (Icc_add_bij _ _ _).image_eq


@[simp]
theorem image_add_const_Ico : (fun x => x + a) '' Ico b c = Ico (b + a) (c + a) :=
  (Ico_add_bij _ _ _).image_eq


@[simp]
theorem image_add_const_Ioc : (fun x => x + a) '' Ioc b c = Ioc (b + a) (c + a) :=
  (Ioc_add_bij _ _ _).image_eq


@[simp]
theorem image_add_const_Ioo : (fun x => x + a) '' Ioo b c = Ioo (b + a) (c + a) :=
  (Ioo_add_bij _ _ _).image_eq


@[simp]
theorem image_const_add_Ici : (fun x => a + x) '' Ici b = Ici (a + b) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Ici b)) (Set.Ici (HAdd.hAdd a b))
  -/
  simp only [add_comm a, image_add_const_Ici]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_add_Ioi : (fun x => a + x) '' Ioi b = Ioi (a + b) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Ioi b)) (Set.Ioi (HAdd.hAdd a b))
  -/
  simp only [add_comm a, image_add_const_Ioi]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_add_Icc : (fun x => a + x) '' Icc b c = Icc (a + b) (a + c) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b c : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Icc b c)) (Set.Icc (HAdd.hAdd a  …
  -/
  simp only [add_comm a, image_add_const_Icc]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_add_Ico : (fun x => a + x) '' Ico b c = Ico (a + b) (a + c) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b c : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Ico b c)) (Set.Ico (HAdd.hAdd a  …
  -/
  simp only [add_comm a, image_add_const_Ico]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_add_Ioc : (fun x => a + x) '' Ioc b c = Ioc (a + b) (a + c) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b c : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Ioc b c)) (Set.Ioc (HAdd.hAdd a  …
  -/
  simp only [add_comm a, image_add_const_Ioc]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_const_add_Ioo : (fun x => a + x) '' Ioo b c = Ioo (a + b) (a + c) := by
  /-
    M : Type u_1
    inst✝¹ : OrderedCancelAddCommMonoid M
    inst✝ : ExistsAddOfLE M
    a b c : M
    ⊢ Eq (Set.image (fun x => HAdd.hAdd a x) (Set.Ioo b c)) (Set.Ioo (HAdd.hAdd a  …
  -/
  simp only [add_comm a, image_add_const_Ioo]
  /-
    🎉 no goals
  -/


